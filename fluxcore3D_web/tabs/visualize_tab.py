"""
tabs/visualize_tab.py — Results NPZ browser, field selector, render button.
"""
from __future__ import annotations
from nicegui import ui


def build_visualize_tab(
    storage: dict,
    on_render=None,
    on_browse_npz=None,
) -> dict:
    """
    Build Visualize tab content.

    Returns refs dict with widgets that need updating externally.
    """
    refs: dict = {}

    # ── Results file ──────────────────────────────────────────────────────────
    with ui.card().classes("w-full bg-neutral-800 border border-neutral-700 gap-2"):
        ui.label("RESULTS FILE").classes(
            "text-xs font-bold tracking-widest text-slate-400 uppercase"
        )
        with ui.row().classes("gap-2 items-center w-full"):
            npz_lbl = ui.label(
                storage.get("vis_npz", "") or "No file loaded"
            ).classes("text-xs text-sky-400 italic flex-1 truncate")
            refs["npz_lbl"] = npz_lbl

            ui.button(
                icon="folder_open", text="Browse…",
                on_click=lambda: on_browse_npz() if on_browse_npz else None,
            ).props("flat dense unelevated").classes(
                "text-slate-300 border border-neutral-600 px-2 py-1 text-xs"
            )

        async def _on_plot_click():
            if on_render:
                coro = on_render()
                if coro is not None:
                    import asyncio as _asyncio
                    if _asyncio.iscoroutine(coro):
                        await coro

        ui.button(
            icon="bar_chart", text="📊  Plot in Viewer",
            on_click=_on_plot_click,
        ).props("unelevated").classes(
            "w-full bg-gradient-to-r from-blue-900 to-emerald-900 "
            "text-white font-semibold py-2 mt-1"
        )

        ui.checkbox(
            "Low quality (faster, less lag)",
            value=storage.get("vis_low_quality", True),
            on_change=lambda e: storage.update({"vis_low_quality": e.value}),
        ).classes("text-xs text-slate-400")

    # ── Field selection ───────────────────────────────────────────────────────
    with ui.card().classes("w-full bg-neutral-800 border border-neutral-700 gap-2"):
        ui.label("FIELD").classes(
            "text-xs font-bold tracking-widest text-slate-400 uppercase"
        )
        with ui.grid(columns=2).classes("gap-x-6 gap-y-1 items-center"):
            ui.label("Scalar field").classes("text-xs text-slate-400")
            ui.select(
                options=["T", "speed", "rho"],
                value=storage.get("vis_field", "T"),
                on_change=lambda e: storage.update({"vis_field": e.value}),
            ).props("dense outlined").classes("w-full")

            ui.label("Colormap").classes("text-xs text-slate-400")
            ui.select(
                options=["inferno","turbo","hot","coolwarm","viridis","jet"],
                value=storage.get("vis_cmap", "inferno"),
                on_change=lambda e: storage.update({"vis_cmap": e.value}),
            ).props("dense outlined").classes("w-full")

    # ── Domains to show ───────────────────────────────────────────────────────
    with ui.card().classes("w-full bg-neutral-800 border border-neutral-700 gap-2"):
        ui.label("DOMAINS TO SHOW").classes(
            "text-xs font-bold tracking-widest text-slate-400 uppercase"
        )

        with ui.row().classes("gap-3 items-center"):
            ui.checkbox(
                "Fluid domain",
                value=storage.get("vis_fluid", False),
                on_change=lambda e: storage.update({"vis_fluid": e.value}),
            ).classes("text-xs text-slate-300")
            ui.label("opacity").classes("text-xs text-slate-400")
            ui.number(
                value=storage.get("vis_fluid_opacity", 0.3),
                min=0.05, max=1.0, step=0.05, format="%.2f",
                on_change=lambda e: storage.update({"vis_fluid_opacity": e.value}),
            ).props("dense outlined").classes("w-20")

        bodies_col = ui.column().classes("w-full gap-0.5")
        refs["bodies_col"] = bodies_col

        with bodies_col:
            _rebuild_body_checks(storage, bodies_col, refs)

        hint = ui.label(
            "Body checkboxes auto-populated when NPZ is loaded"
        ).classes("text-xs text-slate-500 italic")
        refs["vis_hint"] = hint

    # ── Streamlines ───────────────────────────────────────────────────────────
    with ui.card().classes("w-full bg-neutral-800 border border-neutral-700 gap-2"):
        ui.label("STREAMLINES").classes(
            "text-xs font-bold tracking-widest text-slate-400 uppercase"
        )
        with ui.grid(columns=2).classes("gap-x-6 gap-y-1 items-center"):
            ui.label("").classes()  # spacer
            ui.checkbox(
                "Show streamlines",
                value=storage.get("vis_stream", True),
                on_change=lambda e: storage.update({"vis_stream": e.value}),
            ).classes("text-xs text-slate-300")

            ui.label("N seeds").classes("text-xs text-slate-400")
            ui.number(
                value=storage.get("vis_nseeds", 150),
                min=10, max=2000, step=10, format="%.0f",
                on_change=lambda e: storage.update({"vis_nseeds": int(e.value)}),
            ).props("dense outlined").classes("w-full")

            ui.label("Seed mode").classes("text-xs text-slate-400")
            ui.select(
                options=["inlet_plane","line","sphere"],
                value=storage.get("vis_seed_mode","inlet_plane"),
                on_change=lambda e: storage.update({"vis_seed_mode": e.value}),
            ).props("dense outlined").classes("w-full")

    return refs


def _rebuild_body_checks(storage: dict, container: ui.column, refs: dict):
    """Rebuild body checkboxes from vis_bodies dict."""
    container.clear()
    vis = storage.get("vis_bodies", {})
    with container:
        for name, checked in vis.items():
            ui.checkbox(
                name, value=checked,
                on_change=lambda e, n=name: _toggle_body(storage, n, e.value),
            ).classes("text-xs text-slate-300")


def _toggle_body(storage: dict, name: str, value: bool):
    vis = dict(storage.get("vis_bodies", {}))
    vis[name] = value
    storage["vis_bodies"] = vis


def populate_vis_bodies(storage: dict, names: list[str], refs: dict):
    """Called when a new NPZ is loaded to fill the body checkboxes."""
    old = storage.get("vis_bodies", {})
    vis = {n: old.get(n, True) for n in names}
    storage["vis_bodies"] = vis

    col = refs.get("bodies_col")
    if col:
        _rebuild_body_checks(storage, col, refs)
