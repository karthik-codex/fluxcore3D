"""
tabs/solver_tab.py — Flow Solver + Thermal Solver parameter forms.
"""
from __future__ import annotations
from nicegui import ui
from backend.model_io import COLLISION_MODES, OUTLET_MODES


def build_solver_tab(storage: dict):
    """Build Solver tab content. All values bind directly to storage."""

    # ── Flow Solver ───────────────────────────────────────────────────────────
    with ui.card().classes("w-full bg-neutral-800 border border-neutral-700 gap-2"):
        ui.label("FLOW SOLVER").classes(
            "text-xs font-bold tracking-widest text-slate-400 uppercase"
        )
        with ui.grid(columns=2).classes("gap-x-6 gap-y-1 items-center"):
            ui.label("Convergence tol u_ema").classes("text-xs text-slate-400")
            ui.number(
                value=storage.get("tol_u_ema", 0.005),
                min=1e-6, max=1.0, step=0.001, format="%.5f",
                on_change=lambda e: storage.update({"tol_u_ema": e.value}),
            ).props("dense outlined").classes("w-full")

            ui.label("Collision model").classes("text-xs text-slate-400")
            ui.select(
                options=COLLISION_MODES,
                value=storage.get("collision", "mrt_smag"),
                on_change=lambda e: storage.update({"collision": e.value}),
            ).props("dense outlined").classes("w-full")

            ui.label("Outlet BC mode").classes("text-xs text-slate-400")
            ui.select(
                options=OUTLET_MODES,
                value=storage.get("outlet_bc", "convective"),
                on_change=lambda e: storage.update({"outlet_bc": e.value}),
            ).props("dense outlined").classes("w-full")

    # ── Thermal Solver ────────────────────────────────────────────────────────
    with ui.card().classes("w-full bg-neutral-800 border border-neutral-700 gap-2"):
        ui.label("THERMAL SOLVER").classes(
            "text-xs font-bold tracking-widest text-slate-400 uppercase"
        )
        with ui.grid(columns=2).classes("gap-x-6 gap-y-1 items-center"):
            ui.label("Max outer iterations").classes("text-xs text-slate-400")
            ui.number(
                value=storage.get("max_outer", 50000),
                min=100, max=200000, step=1000, format="%.0f",
                on_change=lambda e: storage.update({"max_outer": int(e.value)}),
            ).props("dense outlined").classes("w-full")

            ui.label("dT solid tol").classes("text-xs text-slate-400")
            ui.number(
                value=storage.get("tol_dTs", 0.005),
                min=1e-6, max=1.0, step=0.001, format="%.5f",
                on_change=lambda e: storage.update({"tol_dTs": e.value}),
            ).props("dense outlined").classes("w-full")

            ui.label("dT fluid tol").classes("text-xs text-slate-400")
            ui.number(
                value=storage.get("tol_dTf", 0.005),
                min=1e-6, max=1.0, step=0.001, format="%.5f",
                on_change=lambda e: storage.update({"tol_dTf": e.value}),
            ).props("dense outlined").classes("w-full")

            ui.label("dt_scale_max").classes("text-xs text-slate-400")
            ui.number(
                value=storage.get("dt_scale_max", 100.0),
                min=1.0, max=1000.0, step=5.0, format="%.1f",
                on_change=lambda e: storage.update({"dt_scale_max": e.value}),
            ).props("dense outlined").classes("w-full")

            ui.label("Max MG cycles").classes("text-xs text-slate-400")
            ui.number(
                value=storage.get("max_mg", 15),
                min=1, max=100, step=1, format="%.0f",
                on_change=lambda e: storage.update({"max_mg": int(e.value)}),
            ).props("dense outlined").classes("w-full")
