"""
tabs/domains_tab.py — SOLID DOMAIN → FLUID DOMAIN → VOXELIZATION
Order: Solid Domain (bodies + placement slider) → Fluid Domain (Lx/Ly/Lz + medium) → Voxelization
"""
from __future__ import annotations
from nicegui import ui
from backend.model_io import FLUID_PRESETS
from components.bodies_panel import BodiesPanel

# Placement slider: index 0 = 4.00 (Near Inlet), index 12 = 1.00 (Far)
_PLACEMENT_VALUES = [4.00, 3.75, 3.50, 3.25, 3.00, 2.75, 2.50, 2.25, 2.00, 1.75, 1.50, 1.25, 1.00]
_PLACEMENT_DEFAULT_IDX = 6  # 2.50


def _divisor_to_idx(val: float) -> int:
    return min(range(len(_PLACEMENT_VALUES)), key=lambda i: abs(_PLACEMENT_VALUES[i] - val))


def build_domains_tab(
    storage: dict,
    on_bodies_changed=None,
    on_save_model=None,
    on_load_model=None,
) -> tuple:
    """
    Build the Domains tab.
    Returns (BodiesPanel, domain_refs) where domain_refs has Lx/Ly/Lz input refs + _update_grid_lbl.
    """
    domain_refs: dict = {}

    # ── SOLID DOMAIN ─────────────────────────────────────────────────────────
    with ui.card().classes("w-full bg-neutral-800 border border-neutral-700 gap-2"):
        ui.label("SOLID DOMAIN").classes(
            "text-xs font-bold tracking-widest text-slate-400 uppercase"
        )

        bodies_panel = BodiesPanel(storage, on_change=on_bodies_changed)

        with ui.row().classes("items-center gap-2 mt-1"):
            ui.checkbox(
                "GPU ray tracing",
                value=storage.get("gpu_raytrace", True),
                on_change=lambda e: storage.update({"gpu_raytrace": e.value}),
            ).classes("text-xs text-slate-300")
            ui.label("Checked = CUDA voxeliser (faster)").classes("text-xs text-slate-500 italic")

        # ── Solid Placement slider ────────────────────────────────────────────
        ui.separator().classes("bg-neutral-700 mt-1")
        ui.label("SOLID PLACEMENT").classes(
            "text-xs font-bold tracking-widest text-slate-500 uppercase mt-1"
        )

        cur_idx = _divisor_to_idx(storage.get("j0_divisor", 2.5))

        placement_val_lbl_holder: list = []

        def _on_placement(e):
            idx = int(round(e.value))
            div = _PLACEMENT_VALUES[max(0, min(idx, len(_PLACEMENT_VALUES) - 1))]
            storage.update({"j0_divisor": div})
            if placement_val_lbl_holder:
                placement_val_lbl_holder[0].set_text(f"j0 divisor: {div:.2f}")

        with ui.row().classes("w-full items-center gap-2 px-1 mt-1"):
            ui.label("Near Inlet").classes("text-xs text-slate-500 flex-shrink-0")
            ui.slider(
                min=0, max=len(_PLACEMENT_VALUES) - 1,
                step=1,
                value=cur_idx,
                on_change=_on_placement,
            ).classes("flex-1").props("snap")
            ui.label("Far").classes("text-xs text-slate-500 flex-shrink-0")

        placement_val_lbl = ui.label(
            f"j0 divisor: {_PLACEMENT_VALUES[cur_idx]:.2f}"
        ).classes("text-xs text-slate-400 font-mono text-center w-full")
        placement_val_lbl_holder.append(placement_val_lbl)

        ui.label(
            "Controls offset of solid stack from the inlet along the flow axis."
        ).classes("text-xs text-slate-500 italic")

    # ── FLUID DOMAIN ─────────────────────────────────────────────────────────
    with ui.card().classes("w-full bg-neutral-800 border border-neutral-700 gap-2"):
        ui.label("FLUID DOMAIN").classes(
            "text-xs font-bold tracking-widest text-slate-400 uppercase"
        )

        with ui.grid(columns=2).classes("gap-x-6 gap-y-1 items-center"):
            for dim, key, default in [("Lx (m)", "Lx", 5.0),
                                       ("Ly (m)", "Ly", 1.0),
                                       ("Lz (m)", "Lz", 1.0)]:
                ui.label(dim).classes("text-xs text-slate-400")
                inp = ui.number(
                    label=dim,
                    value=storage.get(key, default),
                    min=0.01, max=20.0, step=0.05, format="%.3f",
                    on_change=lambda e, k=key: (
                        storage.update({k: e.value}),
                        domain_refs.get("_update_grid_lbl", lambda: None)(),
                    ),
                ).props("dense outlined").classes("w-full")
                domain_refs[key] = inp

            ui.label("Fluid medium").classes("text-xs text-slate-400")
            ui.select(
                options=list(FLUID_PRESETS.keys()),
                value=storage.get("fluid", "Air (LBM-scaled)"),
                on_change=lambda e: storage.update({"fluid": e.value}),
            ).props("dense outlined").classes("w-full")

    # ── VOXELIZATION ─────────────────────────────────────────────────────────
    with ui.card().classes("w-full bg-neutral-800 border border-neutral-700 gap-2"):
        ui.label("VOXELIZATION").classes(
            "text-xs font-bold tracking-widest text-slate-400 uppercase"
        )

        grid_lbl = ui.label("---").classes("text-xs text-slate-400 font-mono")
        domain_refs["grid_lbl"] = grid_lbl

        def _update_grid_lbl():
            try:
                dx = storage.get("dx_mm", 10.0)
                nx = round(storage.get("Lx", 5.0) * 1000 / dx)
                ny = round(storage.get("Ly", 1.0) * 1000 / dx)
                nz = round(storage.get("Lz", 1.0) * 1000 / dx)
                grid_lbl.set_text(
                    f"Grid: {nx} × {ny} × {nz}   ({nx*ny*nz/1e6:.2f} M cells)"
                )
            except Exception:
                grid_lbl.set_text("---")

        domain_refs["_update_grid_lbl"] = _update_grid_lbl

        with ui.grid(columns=2).classes("gap-x-6 gap-y-1 items-center"):
            ui.label("Voxel size (mm)").classes("text-xs text-slate-400")
            ui.number(
                value=storage.get("dx_mm", 10.0),
                min=0.1, max=100.0, step=0.5, format="%.1f",
                on_change=lambda e: (
                    storage.update({"dx_mm": e.value}),
                    _update_grid_lbl(),
                ),
            ).props("dense outlined").classes("w-full")

        ui.label(
            "Voxel cell pitch — smaller = finer geometry but more cells"
        ).classes("text-xs text-slate-500 italic")

        _update_grid_lbl()

    return bodies_panel, domain_refs
