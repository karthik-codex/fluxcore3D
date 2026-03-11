"""
tabs/domains_tab.py — Fluid domain spinners + Solid domain bodies panel.
"""
from __future__ import annotations
from nicegui import ui
from backend.model_io import FLUID_PRESETS
from components.bodies_panel import BodiesPanel


def build_domains_tab(
    storage: dict,
    on_bodies_changed=None,
    on_save_model=None,
    on_load_model=None,
) -> BodiesPanel:
    """
    Build the Domains tab content.

    Returns the BodiesPanel instance so callers can call .refresh() on load.
    """

    # ── Fluid Domain ──────────────────────────────────────────────────────────
    with ui.card().classes("w-full bg-neutral-800 border border-neutral-700 gap-2"):
        ui.label("FLUID DOMAIN").classes(
            "text-xs font-bold tracking-widest text-slate-400 uppercase"
        )

        with ui.grid(columns=2).classes("gap-x-6 gap-y-1 items-center"):
            for dim, key, default in [("Lx (m)", "Lx", 5.0),
                                       ("Ly (m)", "Ly", 1.0),
                                       ("Lz (m)", "Lz", 1.0)]:
                ui.label(dim).classes("text-xs text-slate-400")
                ui.number(
                    label=dim, value=storage.get(key, default),
                    min=0.01, max=20.0, step=0.05, format="%.3f",
                    on_change=lambda e, k=key: storage.update({k: e.value}),
                ).props("dense outlined").classes("w-full")

            ui.label("j0 divisor").classes("text-xs text-slate-400")
            ui.number(
                label="j0_divisor",
                value=storage.get("j0_divisor", 2.5),
                min=1.0, max=10.0, step=0.1, format="%.2f",
                on_change=lambda e: storage.update({"j0_divisor": e.value}),
            ).props("dense outlined").classes("w-full")

            ui.label("j0 hint").classes("text-xs text-slate-500 italic col-span-2")
            ui.label(
                "Gap ÷ j0_divisor = body offset from inlet along flow axis"
            ).classes("text-xs text-slate-500 italic col-span-2")

            ui.label("Fluid medium").classes("text-xs text-slate-400")
            ui.select(
                options=list(FLUID_PRESETS.keys()),
                value=storage.get("fluid", "Air (LBM-scaled)"),
                on_change=lambda e: storage.update({"fluid": e.value}),
            ).props("dense outlined").classes("w-full")

    # ── Solid Domain ──────────────────────────────────────────────────────────
    with ui.card().classes("w-full bg-neutral-800 border border-neutral-700 gap-2"):
        ui.label("SOLID DOMAIN").classes(
            "text-xs font-bold tracking-widest text-slate-400 uppercase"
        )

        bodies_panel = BodiesPanel(storage, on_change=on_bodies_changed)

        with ui.row().classes("items-center gap-2 mt-1"):
            gpu_chk = ui.checkbox(
                "GPU ray tracing",
                value=storage.get("gpu_raytrace", True),
                on_change=lambda e: storage.update({"gpu_raytrace": e.value}),
            ).classes("text-xs text-slate-300")
            ui.label(
                "Checked = CUDA voxeliser (faster)"
            ).classes("text-xs text-slate-500 italic")

    return bodies_panel
