"""
components/face_grid.py — FaceGrid: inlet-radio + outlet/wall checkboxes, 3×6 grid.

IMPORTANT NiceGUI rules applied here:
  - Inlet row uses TOGGLE BUTTONS with Python-managed exclusive state (NOT ui.radio).
  - on_change=lambda e: handler(e.value)  — e is GenericEventArguments, e.value is the value.
  - .props(remove="disable") to re-enable, NOT .props("enable").
"""
from __future__ import annotations
from typing import Callable
from nicegui import ui

ALL_FACES  = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]
OPP_FACE   = {"+X":"-X", "-X":"+X", "+Y":"-Y", "-Y":"+Y", "+Z":"-Z", "-Z":"+Z"}

# colour tokens used in the UI
_C_INLET  = "text-emerald-400"
_C_OUTLET = "text-amber-400"
_C_WALL   = "text-sky-400"
_C_HEADER = "text-slate-300"


class FaceGrid:
    """
    Renders a compact 4-row × 7-column grid (header + 3 BC rows).

    Public API
    ----------
    set_flow_dir(flow_dir)   — auto-populate inlet/outlet/walls
    inlet_face               — property → str
    domain_outlets           — property → list[str]
    domain_walls             — property → list[str]
    get_state()              — dict {inlet, outlets, walls}
    set_state(state)         — restore from dict
    """

    def __init__(self, on_change: Callable | None = None):
        self._on_change = on_change
        # Python-side state
        self._inlet   = "-X"
        self._outlets = {f: False for f in ALL_FACES}
        self._walls   = {f: False for f in ALL_FACES}

        # Widget references (not stored in app.storage)
        self._inlet_btns:  dict[str, ui.button]   = {}
        self._outlet_chks: dict[str, ui.checkbox] = {}
        self._wall_chks:   dict[str, ui.checkbox] = {}

        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        with ui.element("div").classes("w-full overflow-x-auto"):
            with ui.grid(columns=7).classes("gap-x-1 gap-y-0.5 items-center"):
                # Row 0 — column headers
                ui.label("").classes("w-14")
                for face in ALL_FACES:
                    ui.label(face).classes(
                        f"font-mono font-bold text-xs text-center {_C_HEADER}"
                    )

                # Row 1 — Inlet (radio-style toggle buttons)
                ui.label("Inlet").classes(f"text-xs font-semibold {_C_INLET}")
                for face in ALL_FACES:
                    btn = (
                        ui.button(
                            icon="radio_button_checked"
                            if face == self._inlet
                            else "radio_button_unchecked",
                            on_click=lambda _e, f=face: self._set_inlet(f),
                        )
                        .props("flat dense size=xs unelevated")
                        .classes("mx-auto w-7 h-7")
                        .tooltip(f"Set inlet = {face}")
                    )
                    self._inlet_btns[face] = btn

                # Row 2 — Outlet checkboxes
                ui.label("Outlet").classes(f"text-xs font-semibold {_C_OUTLET}")
                for face in ALL_FACES:
                    chk = (
                        ui.checkbox(
                            "",
                            value=self._outlets[face],
                            on_change=lambda e, f=face: self._toggle_outlet(f, e.value),
                        )
                        .classes("mx-auto")
                    )
                    self._outlet_chks[face] = chk

                # Row 3 — Wall checkboxes
                ui.label("Wall").classes(f"text-xs font-semibold {_C_WALL}")
                for face in ALL_FACES:
                    chk = (
                        ui.checkbox(
                            "",
                            value=self._walls[face],
                            on_change=lambda e, f=face: self._toggle_wall(f, e.value),
                        )
                        .classes("mx-auto")
                    )
                    self._wall_chks[face] = chk

    # ── Internal state handlers ───────────────────────────────────────────────

    def _set_inlet(self, face: str):
        old_inlet = self._inlet
        self._inlet = face
        # Update button icons (exclusive radio)
        for f, btn in self._inlet_btns.items():
            btn.props(
                "icon=radio_button_checked"
                if f == face
                else "icon=radio_button_unchecked"
            )
        # Disable outlet/wall on the new inlet face; re-enable the old one
        self._outlet_chks[face].value = False
        self._wall_chks[face].value   = False
        self._outlets[face] = False
        self._walls[face]   = False
        self._outlet_chks[face].props("disable")
        self._wall_chks[face].props("disable")
        if old_inlet and old_inlet != face:
            self._outlet_chks[old_inlet].props(remove="disable")
            self._wall_chks[old_inlet].props(remove="disable")
        self._notify()

    def _toggle_outlet(self, face: str, value: bool):
        self._outlets[face] = value
        if value:
            # Mutually exclusive with wall on same face
            self._walls[face] = False
            self._wall_chks[face].value = False
        self._notify()

    def _toggle_wall(self, face: str, value: bool):
        self._walls[face] = value
        if value:
            self._outlets[face] = False
            self._outlet_chks[face].value = False
        self._notify()

    def _notify(self):
        if self._on_change:
            self._on_change(self.get_state())

    # ── Public API ────────────────────────────────────────────────────────────

    def set_flow_dir(self, flow_dir: str):
        """Auto-populate face assignments from a flow direction string."""
        inlet      = OPP_FACE.get(flow_dir, "-X")   # flow enters opposite face
        outlet     = flow_dir                         # flow exits in flow direction
        transverse = [f for f in ALL_FACES if f not in (inlet, outlet)]

        # Update Python state
        self._inlet = inlet
        for f in ALL_FACES:
            self._outlets[f] = (f == outlet)
            self._walls[f]   = (f in transverse)

        # Sync widgets
        for f in ALL_FACES:
            is_inlet = (f == inlet)
            # Inlet toggle
            self._inlet_btns[f].props(
                "icon=radio_button_checked"
                if is_inlet
                else "icon=radio_button_unchecked"
            )
            # Outlet checkbox
            self._outlet_chks[f].value = self._outlets[f]
            if is_inlet:
                self._outlet_chks[f].props("disable")
                self._wall_chks[f].props("disable")
            else:
                self._outlet_chks[f].props(remove="disable")
                self._wall_chks[f].props(remove="disable")
            # Wall checkbox
            self._wall_chks[f].value = self._walls[f]

        self._notify()

    @property
    def inlet_face(self) -> str:
        return self._inlet

    @property
    def domain_outlets(self) -> list[str]:
        return [f for f, v in self._outlets.items() if v]

    @property
    def domain_walls(self) -> list[str]:
        return [f for f, v in self._walls.items() if v]

    def get_state(self) -> dict:
        return dict(
            inlet   = self._inlet,
            outlets = self.domain_outlets,
            walls   = self.domain_walls,
        )

    def set_state(self, state: dict):
        inlet   = state.get("inlet")
        outlets = set(state.get("outlets", []))
        walls   = set(state.get("walls",   []))
        if inlet:
            self._inlet = inlet
        for f in ALL_FACES:
            self._outlets[f] = f in outlets
            self._walls[f]   = f in walls

        for f in ALL_FACES:
            is_inlet = (f == self._inlet)
            self._inlet_btns[f].props(
                "icon=radio_button_checked"
                if is_inlet
                else "icon=radio_button_unchecked"
            )
            self._outlet_chks[f].value = self._outlets[f]
            self._wall_chks[f].value   = self._walls[f]
            if is_inlet:
                self._outlet_chks[f].props("disable")
                self._wall_chks[f].props("disable")
            else:
                self._outlet_chks[f].props(remove="disable")
                self._wall_chks[f].props(remove="disable")

        self._notify()
