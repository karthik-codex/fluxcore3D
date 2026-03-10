"""
components/bodies_panel.py — BodiesPanel: list of solid bodies with add/edit/remove.

NiceGUI rules:
  - Bodies kept as list[dict] in app.storage.user — never widget references.
  - Dialog uses dialog.submit(value) / await dialog pattern.
  - Upload handler uses _read_upload() for version-safe attribute access.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Callable

from nicegui import ui
from backend.model_io import BUILD_DIRS, BODY_COLORS, BODY_ROLES, SOLID_PRESETS

_COLOR_DOT_CSS = {
    "steelblue": "#4682B4", "darkorange": "#FF8C00", "gold": "#FFD700",
    "sienna": "#A0522D", "mediumpurple": "#9370DB", "tomato": "#FF6347",
    "limegreen": "#32CD32", "hotpink": "#FF69B4", "deepskyblue": "#00BFFF",
    "coral": "#FF7F50",
}

_BODY_COUNTER = [0]


def _next_color(idx: int) -> str:
    return BODY_COLORS[idx % len(BODY_COLORS)]


def _make_blank_body(idx: int) -> dict:
    return dict(
        stl_path  = "",
        name      = f"body_{idx+1}",
        build_dir = "+Z",
        material  = "Aluminum (LBM-scaled)",
        color     = _next_color(idx),
        role      = "fluid_base" if idx == 0 else "stack_below",
    )



class BodiesPanel:
    """
    Renders a compact bodies list with toolbar (+ edit delete up down).
    Bodies are stored as list[dict] in storage['bodies'].
    """

    def __init__(self, storage: dict, on_change: Callable[[list], None] | None = None):
        self._storage   = storage
        self._on_change = on_change
        self._selected  = -1

        self._list_col:  ui.column | None = None
        self._btn_edit:  ui.button | None = None
        self._btn_rem:   ui.button | None = None
        self._btn_up:    ui.button | None = None
        self._btn_dn:    ui.button | None = None

        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        with ui.column().classes("w-full gap-1"):
            with ui.row().classes("gap-1 items-center flex-wrap"):
                ui.button(icon="add", on_click=self._add_body).props(
                    "flat dense size=sm unelevated"
                ).classes("text-sky-400").tooltip("Add solid body")

                self._btn_edit = (
                    ui.button(icon="edit", on_click=self._edit_selected)
                    .props("flat dense size=sm unelevated disable")
                    .classes("text-amber-400")
                    .tooltip("Edit selected")
                )
                self._btn_rem = (
                    ui.button(icon="delete", on_click=self._remove_selected)
                    .props("flat dense size=sm unelevated disable")
                    .classes("text-red-400")
                    .tooltip("Remove selected")
                )
                self._btn_up = (
                    ui.button(icon="keyboard_arrow_up", on_click=self._move_up)
                    .props("flat dense size=sm unelevated disable")
                    .classes("text-slate-400")
                    .tooltip("Move up")
                )
                self._btn_dn = (
                    ui.button(icon="keyboard_arrow_down", on_click=self._move_down)
                    .props("flat dense size=sm unelevated disable")
                    .classes("text-slate-400")
                    .tooltip("Move down")
                )
                ui.label("First body = domain reference (fluid_base)").classes(
                    "text-xs text-slate-500 italic ml-2"
                )

            self._list_col = ui.column().classes(
                "w-full gap-0 rounded border border-neutral-700 min-h-20 max-h-44 overflow-y-auto"
            )
            self._refresh_list()

    def _refresh_list(self):
        self._list_col.clear()
        bodies = self._storage.get("bodies", [])
        with self._list_col:
            if not bodies:
                ui.label("No bodies — click + to add").classes(
                    "text-xs text-slate-500 italic px-3 py-2"
                )
                return
            for i, b in enumerate(bodies):
                is_sel  = (i == self._selected)
                dot_col = _COLOR_DOT_CSS.get(b.get("color", "steelblue"), "#4682B4")
                row = (
                    ui.row()
                    .classes(
                        "w-full items-center gap-2 px-2 py-1 cursor-pointer "
                        + ("bg-neutral-700" if is_sel else "hover:bg-neutral-800")
                    )
                    .on("click", lambda _e, idx=i: self._select(idx))
                )
                with row:
                    ui.element("span").style(
                        f"display:inline-block;width:10px;height:10px;"
                        f"border-radius:50%;background:{dot_col};flex-shrink:0"
                    )
                    ui.label(
                        f"{b.get('name','?')}  [{b.get('material','?')}]  · {b.get('role','?')}"
                    ).classes("text-xs text-slate-200 flex-1 truncate")
        self._update_btn_states()

    def _select(self, idx: int):
        self._selected = idx
        self._refresh_list()

    def _update_btn_states(self):
        bodies = self._storage.get("bodies", [])
        n = len(bodies)
        has = 0 <= self._selected < n
        for btn, enabled in (
            (self._btn_edit, has),
            (self._btn_rem,  has and n > 0),
            (self._btn_up,   has and self._selected > 0),
            (self._btn_dn,   has and self._selected < n - 1),
        ):
            if enabled:
                btn.props(remove="disable")
            else:
                btn.props("disable")

    def _notify(self):
        bodies = self._storage.get("bodies", [])
        names  = [b.get("name", "") for b in bodies]
        if self._on_change:
            self._on_change(names)

    # ── Dialog ────────────────────────────────────────────────────────────────

    async def _show_body_dialog(self, body: dict) -> dict | None:
        with ui.dialog() as dlg, ui.card().classes(
            "bg-neutral-800 border border-neutral-600 min-w-xl gap-3"
        ):
            ui.label("Solid Body").classes("text-base font-bold text-slate-200 mb-1")

            # ── STL path: text input + native browse button ───────────────
            # This is a local app — no need to upload; just reference the path on disk.
            ui.label("STL file path").classes("text-xs text-slate-400")
            with ui.row().classes("w-full gap-2 items-center"):
                stl_input = ui.input(
                    placeholder=r"C:\path\to\geometry.stl",
                    value=body.get("stl_path", ""),
                ).props("dense outlined").classes("flex-1 font-mono text-xs")

                def _is_headless() -> bool:
                    """True when running on a server with no display (e.g. RunPod)."""
                    import platform, os as _os
                    if platform.system() != "Windows":
                        return not bool(_os.environ.get("DISPLAY") or _os.environ.get("WAYLAND_DISPLAY"))
                    return False

                def _browse_stl_local():
                    """Native file dialog — falls back to upload if no display available."""
                    try:
                        import tkinter as tk
                        from tkinter import filedialog
                        root = tk.Tk()
                        root.withdraw()
                        root.attributes("-topmost", True)
                        chosen = filedialog.askopenfilename(
                            title="Select STL file",
                            filetypes=[("STL files", "*.stl *.STL"), ("All files", "*.*")],
                        )
                        root.destroy()
                        if chosen:
                            chosen = str(Path(chosen))
                            stl_input.value = chosen
                            body["stl_path"] = chosen
                            if name_input.value.startswith("body_"):
                                stem = Path(chosen).stem
                                name_input.value = stem
                                body["name"]     = stem
                    except Exception as err:
                        # No display available (e.g. RunPod with DISPLAY set but no X server)
                        # silently fall back to the upload dialog
                        import asyncio as _aio
                        _aio.ensure_future(_browse_stl_upload())

                async def _browse_stl_upload():
                    """Upload dialog for headless/cloud environments."""
                    _uploaded = {}

                    with ui.dialog() as upload_dlg, ui.card().classes(
                        "bg-neutral-800 border border-neutral-600 gap-3"
                    ).style("min-width:420px"):
                        ui.label("Upload STL file").classes("text-sm font-bold text-slate-200")

                        _upload_dir = Path(__file__).resolve().parent.parent / "static" / "uploads"
                        _upload_dir.mkdir(parents=True, exist_ok=True)

                        uploader = ui.upload(
                            label="Drop STL here or click to select",
                            auto_upload=True,
                        ).props("accept='.stl,.STL' flat").classes("w-full")

                        status_lbl = ui.label("No file selected").classes("text-xs text-slate-500")

                        with ui.row().classes("justify-end gap-2 w-full mt-1"):
                            ui.button("Cancel", on_click=lambda: upload_dlg.submit(False)).props("flat dense").classes("text-slate-400")
                            ok_btn = ui.button("OK", on_click=lambda: upload_dlg.submit(True)).props("unelevated dense").classes("bg-sky-700 text-white px-4").set_enabled(False)

                        def _handle_upload(e):
                            dest = _upload_dir / e.name
                            with open(dest, "wb") as f:
                                f.write(e.content.read())
                            _uploaded["path"] = str(dest)
                            _uploaded["stem"] = Path(e.name).stem
                            # Hide uploader, show confirmation, enable OK
                            uploader.set_visibility(False)
                            status_lbl.set_text(f"✔  {e.name}  ({dest.stat().st_size // 1024} KB) — ready")
                            status_lbl.classes(replace="text-xs text-emerald-400 font-semibold")
                            ok_btn.set_enabled(True)

                        uploader.on_upload(_handle_upload)

                    result = await upload_dlg
                    if result and _uploaded.get("path"):
                        chosen = _uploaded["path"]
                        stl_input.value = chosen
                        body["stl_path"] = chosen
                        if name_input.value.startswith("body_"):
                            name_input.value = _uploaded["stem"]
                            body["name"] = _uploaded["stem"]
                        ui.notify(f"STL set → {Path(chosen).name}", type="positive")

                import asyncio as _asyncio_bp
                if _is_headless():
                    ui.button(icon="cloud_upload", on_click=_browse_stl_upload).props(
                        "flat dense"
                    ).classes("text-sky-400").tooltip("Upload STL from your computer")
                else:
                    ui.button(icon="folder_open", on_click=_browse_stl_local).props(
                        "flat dense"
                    ).classes("text-sky-400").tooltip("Browse for STL file")

            # Keep body dict in sync whenever the path field changes
            stl_input.on("blur",  lambda _: body.update({"stl_path": stl_input.value.strip()}))
            stl_input.on("keyup", lambda _: body.update({"stl_path": stl_input.value.strip()}))

            ui.separator().classes("bg-neutral-600")

            with ui.grid(columns=2).classes("gap-x-4 gap-y-2 w-full items-center"):
                ui.label("Body name").classes("text-xs text-slate-400")
                name_input = ui.input(
                    value=body.get("name", "body_1")
                ).classes("w-full").props("dense")

                ui.label("Build direction").classes("text-xs text-slate-400")
                build_sel = ui.select(
                    options=BUILD_DIRS, value=body.get("build_dir", "+Z")
                ).classes("w-full").props("dense outlined")

                ui.label("Material").classes("text-xs text-slate-400")
                mat_sel = ui.select(
                    options=list(SOLID_PRESETS.keys()),
                    value=body.get("material", "Aluminum (LBM-scaled)")
                ).classes("w-full").props("dense outlined")

                ui.label("Display color").classes("text-xs text-slate-400")
                col_sel = ui.select(
                    options=BODY_COLORS, value=body.get("color", "steelblue")
                ).classes("w-full").props("dense outlined")

                ui.label("Role").classes("text-xs text-slate-400")
                role_sel = ui.select(
                    options=BODY_ROLES, value=body.get("role", "fluid_base")
                ).classes("w-full").props("dense outlined")

            ui.label(
                "fluid_base = domain-reference body  ·  stack_below = stacked under base"
            ).classes("text-xs text-slate-500 italic")

            with ui.row().classes("justify-end gap-2 mt-2"):
                ui.button("Cancel", on_click=lambda: dlg.submit(None)).props(
                    "flat dense"
                ).classes("text-slate-400")

                def _ok():
                    if not body.get("stl_path"):
                        ui.notify("Select an STL file first.", type="warning")
                        return
                    body["name"]      = name_input.value.strip() or body["name"]
                    body["build_dir"] = build_sel.value
                    body["material"]  = mat_sel.value
                    body["color"]     = col_sel.value
                    body["role"]      = role_sel.value
                    dlg.submit(body)

                ui.button("OK", on_click=_ok).props("dense").classes(
                    "bg-sky-700 text-white"
                )

        return await dlg

    # ── Button handlers ───────────────────────────────────────────────────────

    async def _add_body(self):
        bodies = self._storage.get("bodies", [])
        blank  = _make_blank_body(len(bodies))
        result = await self._show_body_dialog(blank)
        if result:
            bodies.append(result)
            self._storage["bodies"] = bodies
            self._selected = len(bodies) - 1
            self._refresh_list()
            self._notify()

    async def _edit_selected(self):
        bodies = self._storage.get("bodies", [])
        if not (0 <= self._selected < len(bodies)):
            return
        import copy
        body   = copy.deepcopy(bodies[self._selected])
        result = await self._show_body_dialog(body)
        if result:
            bodies[self._selected] = result
            self._storage["bodies"] = bodies
            self._refresh_list()
            self._notify()

    def _remove_selected(self):
        bodies = self._storage.get("bodies", [])
        if not (0 <= self._selected < len(bodies)):
            return
        bodies.pop(self._selected)
        self._storage["bodies"] = bodies
        self._selected = max(0, self._selected - 1) if bodies else -1
        self._refresh_list()
        self._notify()

    def _move_up(self):
        bodies = self._storage.get("bodies", [])
        i = self._selected
        if 0 < i < len(bodies):
            bodies[i], bodies[i-1] = bodies[i-1], bodies[i]
            self._storage["bodies"] = bodies
            self._selected = i - 1
            self._refresh_list()
            self._notify()

    def _move_down(self):
        bodies = self._storage.get("bodies", [])
        i = self._selected
        if 0 <= i < len(bodies) - 1:
            bodies[i], bodies[i+1] = bodies[i+1], bodies[i]
            self._storage["bodies"] = bodies
            self._selected = i + 1
            self._refresh_list()
            self._notify()

    # ── External API ─────────────────────────────────────────────────────────

    def refresh(self):
        self._refresh_list()
