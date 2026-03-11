"""
tabs/bcs_tab.py — Fluid BCs, Solid BCs (surface flux / volumetric), voxelisation.

Key fix: ALL inner functions (_on_face_change, _on_flow_dir, _set_bc, etc.)
are defined BEFORE any widget or callback that references them.
FaceGrid.set_state() is called AFTER _on_face_change is defined.
"""
from __future__ import annotations
from nicegui import ui
from backend.model_io import FLOW_DIRS, FLUX_AXES
from components.face_grid import FaceGrid


def build_bcs_tab(
    storage: dict,
    on_face_change=None,
    body_names: list[str] | None = None,
) -> tuple[FaceGrid, dict]:
    body_names = body_names or []
    refs: dict = {}

    # ── Define ALL callback functions FIRST ───────────────────────────────────
    face_grid_holder: list = []

    def _on_face_change(state: dict):
        storage["face_inlet"]   = state["inlet"]
        storage["face_outlets"] = state["outlets"]
        storage["face_walls"]   = state["walls"]
        if on_face_change:
            on_face_change(state)

    def _on_flow_dir(val: str):
        storage["flow_dir"] = val
        if face_grid_holder:
            face_grid_holder[0].set_flow_dir(val)


    off_page_holder:  list = []
    flux_page_holder: list = []
    vol_page_holder:  list = []
    bc_btns: dict = {}
    bc_type_state = {"value": storage.get("bc_type", "off")}

    def _update_bc_buttons(bct: str):
        for key, btn in bc_btns.items():
            if key == bct:
                btn.classes(replace="bg-sky-700 text-white").props("unelevated dense")
            else:
                btn.classes(replace="bg-neutral-700 text-slate-300").props("unelevated dense")

    def _set_bc(bct: str):
        bc_type_state["value"] = bct
        storage["bc_type"] = bct
        _update_bc_buttons(bct)
        if off_page_holder:  off_page_holder[0].set_visibility(bct == "off")
        if flux_page_holder: flux_page_holder[0].set_visibility(bct == "surface_flux")
        if vol_page_holder:  vol_page_holder[0].set_visibility(bct == "volumetric")

    # ── Fluid BCs card ────────────────────────────────────────────────────────
    with ui.card().classes("w-full bg-neutral-800 border border-neutral-700 gap-2"):
        ui.label("FLUID BCS").classes(
            "text-xs font-bold tracking-widest text-slate-400 uppercase"
        )

        with ui.grid(columns=2).classes("gap-x-6 gap-y-1 items-center"):
            ui.label("Flow direction").classes("text-xs text-slate-400")
            flow_sel = ui.select(
                options=FLOW_DIRS,
                value=storage.get("flow_dir", "+X"),
                on_change=lambda e: _on_flow_dir(e.value),
            ).props("dense outlined").classes("w-full")
            refs["flow_sel"] = flow_sel

            ui.label("Inlet velocity (m/s)").classes("text-xs text-slate-400")
            ui.number(
                value=storage.get("u_in", 1.0),
                min=1e-4, max=1000.0, step=0.01, format="%.4f",
                on_change=lambda e: storage.update({"u_in": e.value}),
            ).props("dense outlined").classes("w-full")

            ui.label("Inlet temperature (C)").classes("text-xs text-slate-400")
            ui.number(
                value=storage.get("t_in_C", 20.0),
                min=-50, max=500, step=0.5, format="%.1f",
                on_change=lambda e: storage.update({"t_in_C": e.value}),
            ).props("dense outlined").classes("w-full")

            ui.label("Ambient temperature (C)").classes("text-xs text-slate-400")
            ui.number(
                value=storage.get("t_amb_C", 20.0),
                min=-50, max=500, step=0.5, format="%.1f",
                on_change=lambda e: storage.update({"t_amb_C": e.value}),
            ).props("dense outlined").classes("w-full")

        ui.label("Face boundary types").classes("text-xs text-slate-500 italic mt-1")

        # _on_face_change already defined above -- safe to pass here
        face_grid = FaceGrid(on_change=_on_face_change)
        face_grid_holder.append(face_grid)
        refs["face_grid"] = face_grid

        # set_state fires _notify -> _on_face_change -- safe now
        face_grid.set_state(dict(
            inlet   = storage.get("face_inlet",   "-X"),
            outlets = storage.get("face_outlets",  ["+X"]),
            walls   = storage.get("face_walls",    ["+Y", "-Y", "+Z", "-Z"]),
        ))

    # ── Solid BCs card ────────────────────────────────────────────────────────
    with ui.card().classes("w-full bg-neutral-800 border border-neutral-700 gap-2"):
        ui.label("SOLID BCS").classes(
            "text-xs font-bold tracking-widest text-slate-400 uppercase"
        )

        with ui.row().classes("gap-2 mb-1"):
            btn_off  = ui.button("Off",             on_click=lambda: _set_bc("off")).props("unelevated dense")
            btn_flux = ui.button("Surface Flux",    on_click=lambda: _set_bc("surface_flux")).props("unelevated dense")
            btn_vol  = ui.button("Volumetric Heat", on_click=lambda: _set_bc("volumetric")).props("unelevated dense")
            bc_btns.update(off=btn_off, surface_flux=btn_flux, volumetric=btn_vol)
            refs.update(btn_off=btn_off, btn_flux=btn_flux, btn_vol=btn_vol)

        off_page = ui.label("No heat source applied.").classes(
            "text-xs text-slate-500 italic py-2"
        )
        off_page_holder.append(off_page)

        with ui.element("div").classes("w-full") as flux_page:
            with ui.grid(columns=2).classes("gap-x-6 gap-y-1 items-center"):
                ui.label("Target solid").classes("text-xs text-slate-400")
                _flux_opts = body_names if body_names else [""]
                _flux_val  = storage.get("flux_solid", "")
                if _flux_val not in _flux_opts:
                    _flux_val = _flux_opts[0]
                flux_solid_sel = ui.select(
                    options=_flux_opts,
                    value=_flux_val,
                    on_change=lambda e: storage.update({"flux_solid": e.value}),
                ).props("dense outlined").classes("w-full")
                refs["flux_solid_sel"] = flux_solid_sel

                ui.label("Flux axis").classes("text-xs text-slate-400")
                ui.select(
                    options=FLUX_AXES,
                    value=storage.get("flux_axis", "-Z"),
                    on_change=lambda e: storage.update({"flux_axis": e.value}),
                ).props("dense outlined").classes("w-full")

                ui.label("Surface L (mm)").classes("text-xs text-slate-400")
                ui.number(
                    value=storage.get("flux_L_mm", 400.0), min=1, max=5000, step=1, format="%.1f",
                    on_change=lambda e: storage.update({"flux_L_mm": e.value}),
                ).props("dense outlined").classes("w-full")

                ui.label("Surface W (mm)").classes("text-xs text-slate-400")
                ui.number(
                    value=storage.get("flux_W_mm", 200.0), min=1, max=5000, step=1, format="%.1f",
                    on_change=lambda e: storage.update({"flux_W_mm": e.value}),
                ).props("dense outlined").classes("w-full")

                ui.label("Heat flux q (W/m2)").classes("text-xs text-slate-400")
                ui.number(
                    value=storage.get("flux_q", 1800.0), min=0, max=1e7, step=100, format="%.1f",
                    on_change=lambda e: storage.update({"flux_q": e.value}),
                ).props("dense outlined").classes("w-full")

                ui.label("Auto-centre").classes("text-xs text-slate-400")
                ui.checkbox(
                    "Auto-centre on solid centroid",
                    value=storage.get("flux_autocenter", True),
                    on_change=lambda e: storage.update({"flux_autocenter": e.value}),
                ).classes("text-xs")
        flux_page_holder.append(flux_page)

        with ui.element("div").classes("w-full") as vol_page:
            with ui.grid(columns=2).classes("gap-x-6 gap-y-1 items-center"):
                ui.label("Target solid").classes("text-xs text-slate-400")
                _vol_opts = body_names if body_names else [""]
                _vol_val  = storage.get("vol_solid", "")
                if _vol_val not in _vol_opts:
                    _vol_val = _vol_opts[0]
                vol_solid_sel = ui.select(
                    options=_vol_opts,
                    value=_vol_val,
                    on_change=lambda e: storage.update({"vol_solid": e.value}),
                ).props("dense outlined").classes("w-full")
                refs["vol_solid_sel"] = vol_solid_sel

                ui.label("Total power (W)").classes("text-xs text-slate-400")
                ui.number(
                    value=storage.get("vol_Q", 50.0), min=0, max=100000, step=1, format="%.1f",
                    on_change=lambda e: storage.update({"vol_Q": e.value}),
                ).props("dense outlined").classes("w-full")
        vol_page_holder.append(vol_page)

        # Apply initial BC visibility -- all page holders are populated now
        _set_bc(storage.get("bc_type", "off"))

    return face_grid, refs


def update_body_dropdowns(refs: dict, names: list[str], storage: dict):
    """Called from app.py whenever bodies list changes."""
    for key in ("flux_solid_sel", "vol_solid_sel"):
        sel = refs.get(key)
        if sel is None:
            continue
        sel.options = names
        storage_key = "flux_solid" if "flux" in key else "vol_solid"
        cur = storage.get(storage_key, "")
        sel.value = cur if cur in names else (names[0] if names else "")
