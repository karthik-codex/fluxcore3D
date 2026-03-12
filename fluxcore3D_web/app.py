"""
app.py — FluxCore3D NiceGUI web app entry point.
"""
from __future__ import annotations
import sys
import os
import json
import time
import tempfile
import base64
import asyncio
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_APP_DIR))
sys.path.insert(0, str(_APP_DIR.parent))

from nicegui import ui, app, run

from backend.state     import get_storage
from backend.model_io  import (
    save_model_file, load_model_file, restore_storage_from_model,
    build_state_dict,
)
from backend.runner    import (
    _sim_state, _preview_state,
    launch_sim_thread, launch_preview_thread,
    abort_sim, reset_sim_state,
    render_results_b64,
    export_results_gltf,
    HAS_SIM as _RUNNER_HAS_SIM,
    SIM_IMPORT_ERROR as _RUNNER_SIM_ERR,
)
from tabs.domains_tab   import build_domains_tab
from tabs.bcs_tab       import build_bcs_tab, update_body_dropdowns
from tabs.solver_tab    import build_solver_tab
from tabs.visualize_tab import build_visualize_tab, populate_vis_bodies

app.add_static_files("/static", str(_APP_DIR / "static"))


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE
# ─────────────────────────────────────────────────────────────────────────────

@ui.page("/")
async def main_page():
    storage = get_storage()

    _msg = storage.pop("_load_msg", None)
    if _msg:
        ui.timer(0.3, lambda: ui.notify(f"📂  {_msg}", type="positive"), once=True)

    def _show_sim_status():
        log = ui_state.get("log_area")
        if log is None: return
        if _RUNNER_HAS_SIM:
            log.push("✅  Sim modules loaded — CUDA extensions ready.")
        else:
            log.push(f"⚠️  Sim modules unavailable: {_RUNNER_SIM_ERR}")
    ui.timer(0.6, _show_sim_status, once=True)

    async def _restore_results_on_load():
        npz = storage.get("result_npz", "")
        if npz and Path(npz).exists():
            log = ui_state.get("log_area")
            if log: log.push(f"📂  Restoring last results: {Path(npz).name}")
            await _load_npz_into_viewer(npz, storage, ui_state)
    ui.timer(1.0, _restore_results_on_load, once=True)

    ui_state = {
        "sim_running":     False,
        "prev_log_len":    0,
        "prev_log_len_sim":0,
        "image_panel":     None,
        "log_area":        None,
        "progress_bar":    None,
        "status_lbl":      None,
        "run_btn":         None,
        "preview_btn":     None,
        "bodies_panel_ref":None,
        "bcs_refs":        {},
        "vis_refs":        {},
        "face_grid_ref":   None,
        "diag_dialog":     None,
    }

    ui.add_head_html(
        '<script>'
        '(function(){'
        'var _orig=HTMLCanvasElement.prototype.getContext;'
        'HTMLCanvasElement.prototype.getContext=function(t,a){'
        'if(t==="webgl"||t==="webgl2")a=Object.assign({},a,{preserveDrawingBuffer:true});'
        'return _orig.call(this,t,a);'
        '};'
        '})();'
        '</script>'
        '<link rel="stylesheet" href="/static/theme.css">'
        '<link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">'
    )

    with ui.row().classes(
        "w-full items-center gap-3 px-4 py-0 border-b border-neutral-700"
    ).style("background:#1e1e1e"):
        ui.html('<svg width="356" height="72" viewBox="0 0 178 36" fill="none" xmlns="http://www.w3.org/2000/svg"><defs><linearGradient id="fc3d_i" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stop-color="#000004"/><stop offset="13%" stop-color="#1B0C41"/><stop offset="26%" stop-color="#4A0C6B"/><stop offset="38%" stop-color="#781C6D"/><stop offset="50%" stop-color="#A52C60"/><stop offset="62%" stop-color="#CF4446"/><stop offset="74%" stop-color="#ED6925"/><stop offset="86%" stop-color="#FB9A06"/><stop offset="93%" stop-color="#F7D13D"/><stop offset="100%" stop-color="#FCFFA4"/></linearGradient><linearGradient id="fc3d_bg" x1="0" y1="1" x2="1" y2="0"><stop offset="0%" stop-color="#0072FF"/><stop offset="100%" stop-color="#00C6FF"/></linearGradient><linearGradient id="fc3d_wm" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stop-color="#00C6FF"/><stop offset="100%" stop-color="#00FFA3"/></linearGradient><filter id="fc3d_glow"><feGaussianBlur stdDeviation="1.1" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter><clipPath id="fc3d_cc"><rect x="5" y="5" width="24" height="24" rx="1.5"/></clipPath></defs><rect x="5" y="5" width="24" height="24" rx="1.5" fill="#080c14" stroke="#1a2440" stroke-width="1"/><rect x="5" y="5" width="24" height="24" rx="1.5" fill="url(#fc3d_bg)" opacity="0.08" clip-path="url(#fc3d_cc)"/><path d="M0,14 C5,14 7,11 11,11 S17,14 22,14 S27,17 34,17" stroke="url(#fc3d_i)" stroke-width="1.3" fill="none" filter="url(#fc3d_glow)" opacity="1"/><path d="M0,18 C5,18 7,15 11,15 S17,18 22,18 S27,21 34,21" stroke="url(#fc3d_i)" stroke-width="0.9" fill="none" opacity="0.65"/><path d="M0,22 C5,22 7,19 11,19 S17,22 22,22 S27,25 34,25" stroke="url(#fc3d_i)" stroke-width="0.6" fill="none" opacity="0.35"/><line x1="11" y1="5" x2="11" y2="2" stroke="#1e3a4a" stroke-width="1.3" stroke-linecap="round"/><line x1="17" y1="5" x2="17" y2="2" stroke="#1e3a4a" stroke-width="1.3" stroke-linecap="round"/><line x1="23" y1="5" x2="23" y2="2" stroke="#1e3a4a" stroke-width="1.3" stroke-linecap="round"/><line x1="11" y1="29" x2="11" y2="32" stroke="#1e3a4a" stroke-width="1.3" stroke-linecap="round"/><line x1="17" y1="29" x2="17" y2="32" stroke="#1e3a4a" stroke-width="1.3" stroke-linecap="round"/><line x1="23" y1="29" x2="23" y2="32" stroke="#1e3a4a" stroke-width="1.3" stroke-linecap="round"/><line x1="5" y1="12" x2="2" y2="12" stroke="#1e3a4a" stroke-width="1.3" stroke-linecap="round"/><line x1="5" y1="18" x2="2" y2="18" stroke="#1e3a4a" stroke-width="1.3" stroke-linecap="round"/><line x1="5" y1="24" x2="2" y2="24" stroke="#1e3a4a" stroke-width="1.3" stroke-linecap="round"/><line x1="29" y1="12" x2="32" y2="12" stroke="#1e3a4a" stroke-width="1.3" stroke-linecap="round"/><line x1="29" y1="18" x2="32" y2="18" stroke="#1e3a4a" stroke-width="1.3" stroke-linecap="round"/><line x1="29" y1="24" x2="32" y2="24" stroke="#1e3a4a" stroke-width="1.3" stroke-linecap="round"/><text x="40" y="15" font-family="Courier New,monospace" font-size="10" font-weight="700" letter-spacing="3" fill="url(#fc3d_wm)">FLUX</text><text x="40" y="28" font-family="Courier New,monospace" font-size="10" font-weight="700" letter-spacing="3" fill="#e2e8f0">CORE<tspan fill="#00FFA3" letter-spacing="1.5">3D</tspan></text></svg>').classes("flex-shrink-0")
        ui.label("Thermal Intelligence for Electronics Cooling").classes(
            "text-large text-slate-500 tracking-widest font-mono"
        )
        ui.space()
        ui.label("Project:").classes("text-xs text-slate-400 flex-shrink-0")
        proj_input = ui.input(
            placeholder="project_name",
            value=storage.get("project_name", "cht_result"),
            on_change=lambda e: storage.update({"project_name": e.value}),
        ).props("dense outlined").classes("w-36 text-sm flex-shrink-0")
        ui_state["proj_input"] = proj_input
        ui.button(
            icon="add_circle_outline", text="New",
            on_click=lambda: _new_model_dialog(storage, ui_state),
        ).props("flat dense unelevated").classes(
            "text-emerald-400 border border-neutral-600 px-2 text-xs flex-shrink-0"
        )
        ui.button(
            icon="folder_open", text="Load",
            on_click=lambda: _load_model_dialog(storage, ui_state),
        ).props("flat dense unelevated").classes(
            "text-slate-300 border border-neutral-600 px-2 text-xs flex-shrink-0"
        )
        ui.button(
            icon="save", text="Save",
            on_click=lambda: _save_model_dialog(storage, ui_state),
        ).props("flat dense unelevated").classes(
            "text-slate-300 border border-neutral-600 px-2 text-xs flex-shrink-0"
        )

    with ui.splitter(value=25).classes("w-full flex-1").style(
        "height: calc(100vh - 52px)"
    ) as splitter:
        with splitter.before:
            _build_control_panel(storage, ui_state)
        with splitter.after:
            _build_viewer_panel(storage, ui_state)

    # ── Polling timers — entire body wrapped in try/except RuntimeError ────────
    async def _poll_sim():
        try:
            if not _sim_state["running"] and not _sim_state["done"]:
                return

            log: ui.log = ui_state.get("log_area")
            prev_len = ui_state["prev_log_len_sim"]
            msgs = _sim_state["log"]
            for msg in msgs[prev_len:]:
                if log:
                    log.push(msg)
            ui_state["prev_log_len_sim"] = len(msgs)

            pb = ui_state.get("progress_bar")
            if pb:
                pb.value = _sim_state["pct"] / 100.0

            sl = ui_state.get("status_lbl")
            if sl and msgs:
                sl.set_text(msgs[-1][:70])

            if _sim_state["done"]:
                _sim_timer.deactivate()
                ui_state["sim_running"] = False
                _set_run_btn_idle(ui_state)

                if _sim_state["error"]:
                    ui.notify(f"Simulation failed: {_sim_state['error'][:120]}",
                              type="negative", timeout=6000)
                    reset_sim_state()
                elif _sim_state["result"]:
                    npz = _sim_state["result"]
                    storage["result_npz"]      = npz
                    storage["result_flow_dir"] = storage.get("flow_dir", "+X")
                    ui.notify(f"✅  Done → {Path(npz).name}", type="positive")
                    reset_sim_state()
                    await _load_npz_into_viewer(npz, storage, ui_state)
                    _show_diagnostics(npz, ui_state)
                else:
                    reset_sim_state()

        except RuntimeError:
            _sim_timer.deactivate()
            return

    _sim_timer = ui.timer(0.5, _poll_sim, active=False)

    async def _poll_preview():
        try:
            if not _preview_state["running"] and not _preview_state["done"]:
                return

            log = ui_state.get("log_area")
            prev_len = ui_state.get("prev_log_len", 0)
            msgs = _preview_state["log"]
            for msg in msgs[prev_len:]:
                if log: log.push(msg)
            ui_state["prev_log_len"] = len(msgs)

            if _preview_state["done"]:
                _prev_timer.deactivate()
                pb = ui_state.get("progress_bar")
                if pb: pb.value = 0
                sl = ui_state.get("status_lbl")
                if _preview_state["error"]:
                    ui.notify(f"Preview failed: {_preview_state['error'][:200]}",
                              type="negative", timeout=8000)
                    if sl: sl.set_text("Preview failed.")
                else:
                    import time as _time
                    _ts = int(_time.time())
                    for _bm in (_preview_state.get("body_meta") or []):
                        if "gltf_url" in _bm:
                            base = _bm["gltf_url"].split("?")[0]
                            _bm["gltf_url"] = f"{base}?v={_ts}"
                    if ui_state.get("viewer_mode", "domain") == "domain":
                        r = ui_state.get("scene_refresh")
                        if r: r()
                    if sl: sl.set_text("Preview ready.")
                    ui.notify("Preview ready.", type="positive", timeout=2000)
                    info = ui_state.get("info_lbl")
                    if info:
                        gm = _preview_state.get("grid_mm") or [0,0,0]
                        fd = _preview_state.get("flow_dir", "?")
                        info.set_text(f"Domain {gm[0]:.0f}×{gm[1]:.0f}×{gm[2]:.0f} mm  flow {fd}  |  {len(_preview_state.get('body_meta') or [])} bodies")
                btn = ui_state.get("preview_btn")
                if btn: btn.props(remove="disable")

        except RuntimeError:
            _prev_timer.deactivate()
            return

    _prev_timer = ui.timer(0.4, _poll_preview, active=False)

    ui_state["_sim_timer"]  = _sim_timer
    ui_state["_prev_timer"] = _prev_timer
    ui_state["_storage"]    = storage


# ─────────────────────────────────────────────────────────────────────────────
#  Control panel (left)
# ─────────────────────────────────────────────────────────────────────────────

def _build_control_panel(storage: dict, ui_state: dict):
    with ui.column().classes(
        "h-full overflow-y-auto bg-neutral-900 border-r border-neutral-700"
    ).style("width:100%; max-width:100%; padding:0"):

        with ui.tabs().classes("w-full").props("dense") as ctrl_tabs:
            t_dom = ui.tab("Domains",   icon="layers")
            t_bc  = ui.tab("BCs",       icon="water_drop")
            t_sol = ui.tab("Solver",    icon="settings")
            t_vis = ui.tab("Visualize", icon="bar_chart")

        with ui.tab_panels(ctrl_tabs, value=t_dom).classes(
            "w-full flex-1 overflow-y-auto"
        ).style("background:#252526"):

            with ui.tab_panel(t_dom).classes("gap-3 flex flex-col"):
                bp, dom_refs = build_domains_tab(
                    storage,
                    on_bodies_changed=lambda names: _on_bodies_changed(names, storage, ui_state),
                    on_save_model=lambda: _save_model_dialog(storage, ui_state),
                    on_load_model=lambda: _load_model_dialog(storage, ui_state),
                )
                ui_state["bodies_panel_ref"] = bp
                ui_state["domain_refs"] = dom_refs

            with ui.tab_panel(t_bc).classes("gap-3 flex flex-col"):
                face_grid, bcs_refs = build_bcs_tab(
                    storage,
                    body_names=[b.get("name","") for b in storage.get("bodies",[])],
                )
                ui_state["face_grid_ref"] = face_grid
                ui_state["bcs_refs"]      = bcs_refs

            with ui.tab_panel(t_sol).classes("gap-3 flex flex-col"):
                build_solver_tab(storage)

            with ui.tab_panel(t_vis).classes("gap-3 flex flex-col"):
                vis_refs = build_visualize_tab(
                    storage,
                    on_render=lambda: _launch_viz(storage, ui_state),
                    on_browse_npz=lambda: _browse_npz_dialog(storage, ui_state),
                )
                ui_state["vis_refs"] = vis_refs

        with ui.column().classes(
            "w-full gap-1 p-3 border-t border-neutral-700 bg-neutral-900"
        ):
            pb = ui.linear_progress(value=0.0, size="4px").props(
                "instant-feedback color=blue"
            ).classes("w-full")
            ui_state["progress_bar"] = pb

            sl = ui.label("Ready").classes("fc3d-status")
            ui_state["status_lbl"] = sl

            with ui.row().classes("gap-2 w-full"):
                prev_btn = (
                    ui.button(
                        icon="visibility", text="Preview",
                        on_click=lambda: _on_preview_click(storage, ui_state),
                    )
                    .props("unelevated")
                    .classes("flex-1 bg-neutral-700 text-slate-200 py-2")
                    .props("disable" if not storage.get("bodies") else "")
                )
                ui_state["preview_btn"] = prev_btn

                run_btn = (
                    ui.button(
                        icon="play_arrow", text="Run Model",
                        on_click=lambda: _on_run_click(storage, ui_state),
                    )
                    .props("unelevated")
                    .classes(
                        "flex-[1.5] text-white py-2 font-semibold "
                        "bg-gradient-to-r from-blue-900 to-emerald-900"
                    )
                )
                ui_state["run_btn"] = run_btn

            log_area = ui.log(max_lines=200).classes(
                "fc3d-log w-full"
            ).style("height:90px;min-height:70px;max-height:110px")
            ui_state["log_area"] = log_area


# ─────────────────────────────────────────────────────────────────────────────
#  Viewer panel (right)
# ─────────────────────────────────────────────────────────────────────────────

def _show_viewer_mode(ui_state: dict, mode: str):
    ui_state["viewer_mode"] = mode
    db = ui_state.get("dom_btn")
    rb = ui_state.get("res_btn")
    il = ui_state.get("info_lbl")
    rl = ui_state.get("res_lbl")
    if db: db.classes(
        add="text-sky-400 border-b-2 border-sky-500" if mode=="domain" else "text-slate-500",
        remove="text-slate-500" if mode=="domain" else "text-sky-400 border-b-2 border-sky-500")
    if rb: rb.classes(
        add="text-sky-400 border-b-2 border-sky-500" if mode=="results" else "text-slate-500",
        remove="text-slate-500" if mode=="results" else "text-sky-400 border-b-2 border-sky-500")
    if il: il.set_visibility(mode=="domain")
    if rl: rl.set_visibility(mode=="results")
    if mode == "results":
        pt = ui_state.get("_prev_timer")
        if pt: pt.deactivate()
    r = ui_state.get("scene_refresh")
    if r: r()


def _build_viewer_panel(storage: dict, ui_state: dict):
    ui_state.setdefault("viewer_mode", "domain")

    with ui.column().classes("h-full w-full gap-0 bg-neutral-900").style("overflow:hidden"):

        with ui.row().classes("w-full items-center gap-0 border-b border-neutral-800").style("flex-shrink:0"):
            dom_btn = ui.button("⬡  Domain Preview",
                on_click=lambda: _show_viewer_mode(ui_state, "domain")
            ).props("flat unelevated").classes(
                "text-xs font-semibold px-4 py-2 text-sky-400 border-b-2 border-sky-500")
            res_btn = ui.button("🌡  Results",
                on_click=lambda: _show_viewer_mode(ui_state, "results")
            ).props("flat unelevated").classes(
                "text-xs font-semibold px-4 py-2 text-slate-500")
            ui_state["dom_btn"] = dom_btn
            ui_state["res_btn"] = res_btn
            ui.space()
            ui.button(icon="folder_open", text="Load Results…",
                on_click=lambda: _load_results_dialog(storage, ui_state)
            ).props("flat dense unelevated").classes(
                "text-slate-300 text-xs border border-neutral-700 px-2 py-1 mr-1")
            full_btn = ui.button(icon="open_in_full", text="PyVista",
                on_click=lambda: _launch_full_viewer(storage)
            ).props("flat dense unelevated disable").classes(
                "text-sky-400 text-xs border border-neutral-700 px-2 py-1 mr-2")
            ui_state["full_viewer_btn"] = full_btn

        with ui.row().classes("w-full items-center px-3 pt-1 gap-2").style("flex-shrink:0"):
            info_lbl = ui.label("Add bodies → click Preview to load here.").classes(
                "text-xs text-slate-500 flex-1")
            ui_state["info_lbl"] = info_lbl
            res_lbl = ui.label("").classes("text-xs text-slate-500 flex-1")
            res_lbl.set_visibility(False)
            ui_state["res_lbl"] = res_lbl

            async def _snapshot():
                mode = ui_state.get("viewer_mode", "domain")
                fname = f"fluxcore3d_{'preview' if mode=='domain' else 'results'}.png"
                await ui.run_javascript(f"""
                    (function() {{
                        const canvases = document.querySelectorAll('canvas');
                        if (!canvases.length) {{ alert('No canvas found'); return; }}
                        let best = canvases[0];
                        for (const c of canvases) {{
                            if (c.width * c.height > best.width * best.height) best = c;
                        }}
                        const link = document.createElement('a');
                        link.download = '{fname}';
                        link.href = best.toDataURL('image/png');
                        link.click();
                    }})();
                """)

            ui.button(icon="photo_camera", text="Snapshot",
                on_click=_snapshot,
            ).props("flat dense unelevated").classes(
                "text-slate-400 text-xs border border-neutral-700 px-2 py-1"
            ).tooltip("Save viewport as PNG")

        @ui.refreshable
        def _scene_widget():
            mode      = ui_state.get("viewer_mode", "domain")
            body_meta = _preview_state.get("body_meta") or []
            grid_mm   = _preview_state.get("grid_mm")   or [100, 100, 100]
            flow_d    = _preview_state.get("flow_dir")  or "+X"
            W, D, H   = grid_mm
            res_meta  = ui_state.get("result_meta") or []

            with ui.row().classes("w-full gap-0 relative").style("flex:1; min-height:0"):
                sc = ui.scene(
                    width=900, height=600,
                    background_color="#0A0E17",
                    grid=False,
                ).classes("w-full").style(
                    "flex:1; height:100% !important; min-height:0"
                )

                with sc:
                    if mode == "domain":
                        if not body_meta:
                            sc.axes_helper(length=5)
                        else:
                            sc.spot_light(color="#ffffff", intensity=0.8,
                                          distance=0, angle=1.2, penumbra=0.1, decay=0.5
                                          ).move(W*0.5, D*2.0, H*2.0)
                            sc.spot_light(color="#ffffff", intensity=0.4,
                                          distance=0, angle=1.5, penumbra=0.2, decay=0.5
                                          ).move(W*0.5, -D*1.5, H*0.5)
                            _draw_box_wireframe(sc, W, D, H)
                            _draw_flow_arrow(sc, W, D, H, flow_d)
                            for bm in body_meta:
                                url = bm.get("gltf_url")
                                if url:
                                    hx = _css_to_hex(bm.get("color", "steelblue"))
                                    sc.gltf(url).material(color=hx, opacity=1.0)
                            sc.axes_helper(length=max(W,D,H)*0.12)
                            sc.move_camera(
                                x=W*0.5, y=-D*1.8, z=H*1.4,
                                look_at_x=W*0.5, look_at_y=D*0.5, look_at_z=H*0.5,
                            )
                    else:
                        if not res_meta:
                            sc.axes_helper(length=5)
                        else:
                            sc.spot_light(color="#ffffff", intensity=0.6,
                                          distance=0, angle=1.5, penumbra=0.2, decay=0.5
                                          ).move(100, -150, 150)
                            sc.spot_light(color="#ffffff", intensity=0.3,
                                          distance=0, angle=1.5, penumbra=0.2, decay=0.5
                                          ).move(-100, 150, 50)
                            for rm in res_meta:
                                url = rm.get("gltf_url")
                                if url:
                                    sc.gltf(url)
                            sc.axes_helper(length=50)
                            _rg = ui_state.get("result_grid_size") or [100, 100, 100]
                            rx, ry, rz = _rg
                            sc.move_camera(
                                x=rx*0.5, y=-ry*2.0, z=rz*1.5,
                                look_at_x=rx*0.5, look_at_y=ry*0.5, look_at_z=rz*0.3,
                            )

                # ── Overlays ──────────────────────────────────────────────────
                legend = body_meta if mode == "domain" else (ui_state.get("result_legend") or [])
                cbar   = ui_state.get("result_cbar") if mode == "results" else None

                if legend:
                    with ui.column().classes(
                        "absolute top-2 right-2 bg-black bg-opacity-70 "
                        "rounded px-3 py-2 gap-1"
                    ).style("pointer-events:none; min-width:140px"):
                        ui.label("Bodies" if mode=="domain" else "Temperature").classes(
                            "text-xs font-bold text-slate-300 mb-1")
                        for bm in legend:
                            hx = _css_to_hex(bm.get("color", "steelblue"))
                            with ui.row().classes("items-center gap-2"):
                                ui.element("div").style(
                                    f"width:12px;height:12px;border-radius:2px;"
                                    f"background:{hx};flex-shrink:0")
                                ui.label(bm.get("name","")).classes("text-xs text-slate-300")

                if cbar:
                    ticks = cbar.get("ticks", [])
                    # FIX: use units from cbar dict, not hardcoded °C
                    _u = cbar.get("units", "°C")
                    _u_str = f" [{_u}]" if _u else ""
                    with ui.column().classes(
                        "absolute bottom-6 left-3 gap-0"
                    ).style("pointer-events:none"):
                        ui.label(f"{cbar.get('field','T')}{_u_str}").classes(
                            "text-xs font-bold text-slate-300 mb-1")
                        with ui.row().classes("gap-1 items-stretch"):
                            grad_colors = ",".join(t["color"] for t in ticks)
                            ui.element("div").style(
                                f"width:14px; height:120px; border-radius:3px;"
                                f"background:linear-gradient(to top, {grad_colors});"
                                f"flex-shrink:0"
                            )
                            with ui.column().classes("justify-between gap-0").style("height:120px"):
                                for tick in reversed(ticks):
                                    ui.label(tick["label"]).classes(
                                        "text-xs text-slate-300 leading-none")

        _scene_widget()
        ui_state["scene_refresh"] = _scene_widget.refresh
        ui_state["domain_scene"]  = None
        ui_state["results_scene"] = None
        ui_state["domain_scene_placeholder"]  = ui.label("").classes("hidden")
        ui_state["results_scene_placeholder"] = ui.label("").classes("hidden")

        with ui.expansion("Terminal Output", icon="terminal").classes(
            "w-full border-t border-neutral-700 bg-neutral-950"
        ).style("flex-shrink:0") as _term_expansion:
            _term_expansion.props("dense")
            with ui.row().classes("w-full justify-end px-1 py-0"):
                def _clear_term():
                    term_log.clear()
                    ui_state["_term_prev_len"] = len(_preview_state.get("log", []))
                    ui_state["_term_sim_len"]  = len(_sim_state.get("log", []))
                ui.button(icon="delete_sweep", text="Clear",
                    on_click=_clear_term,
                ).props("flat dense").classes("text-slate-500 text-xs")

            term_log = ui.log(max_lines=400).classes(
                "w-full font-mono text-xs text-emerald-300 bg-neutral-950"
            ).style(
                "height:180px; white-space:pre; overflow-y:auto;"
                "border:none; padding:4px 8px"
            )
            ui_state["term_log"] = term_log

        def _poll_stdout():
            try:
                tlog = ui_state.get("term_log")
                if tlog is None: return
                prev_len = ui_state.get("_term_prev_len", 0)
                prev_msgs = _preview_state.get("log", [])
                for line in prev_msgs[prev_len:]:
                    tlog.push(f"[PREVIEW] {line}")
                ui_state["_term_prev_len"] = len(prev_msgs)
                sim_len = ui_state.get("_term_sim_len", 0)
                sim_msgs = _sim_state.get("term_log", [])
                for line in sim_msgs[sim_len:]:
                    tlog.push(line)
                ui_state["_term_sim_len"] = len(sim_msgs)
            except RuntimeError:
                pass  # parent slot deleted — page navigated away

        ui.timer(0.3, _poll_stdout)


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

_COLOUR_HEX = {
    "steelblue":"#4682B4","darkorange":"#FF8C00","mediumseagreen":"#3CB371",
    "mediumpurple":"#9370DB","tomato":"#FF6347","gold":"#FFD700",
    "hotpink":"#FF69B4","limegreen":"#32CD32","deepskyblue":"#00BFFF",
    "coral":"#FF7F50","sienna":"#A0522D",
}

def _css_to_hex(c: str) -> str:
    return _COLOUR_HEX.get(c, c) if not c.startswith("#") else c


def _draw_box_wireframe(scene, W, D, H):
    corners = [
        ([0,0,0],[W,0,0]), ([0,D,0],[W,D,0]), ([0,0,H],[W,0,H]), ([0,D,H],[W,D,H]),
        ([0,0,0],[0,D,0]), ([W,0,0],[W,D,0]), ([0,0,H],[0,D,H]), ([W,0,H],[W,D,H]),
        ([0,0,0],[0,0,H]), ([W,0,0],[W,0,H]), ([0,D,0],[0,D,H]), ([W,D,0],[W,D,H]),
    ]
    for s, e in corners:
        scene.line(s, e).material(color="#2A3A5A")


def _draw_flow_arrow(scene, W, D, H, flow_d):
    _AX = {
        "+X": ([0,   D/2, H/2], [W*0.3, D/2, H/2]),
        "-X": ([W,   D/2, H/2], [W*0.7, D/2, H/2]),
        "+Y": ([W/2, 0,   H/2], [W/2, D*0.3, H/2]),
        "-Y": ([W/2, D,   H/2], [W/2, D*0.7, H/2]),
        "+Z": ([W/2, D/2, 0  ], [W/2, D/2, H*0.3]),
        "-Z": ([W/2, D/2, H  ], [W/2, D/2, H*0.7]),
    }
    pts = _AX.get(flow_d)
    if pts:
        scene.line(pts[0], pts[1]).material(color="#00CCFF")


def _set_run_btn_idle(ui_state: dict):
    btn = ui_state.get("run_btn")
    if btn:
        btn.icon  = "play_arrow"
        btn._props["label"] = "Run Model"
        btn.classes(replace=(
            "flex-[1.5] text-white py-2 font-semibold "
            "bg-gradient-to-r from-blue-900 to-emerald-900"
        ))
        btn.update()


def _set_run_btn_running(ui_state: dict):
    btn = ui_state.get("run_btn")
    if btn:
        btn.icon = "stop"
        btn._props["label"] = "Stop"
        btn.classes(replace=(
            "flex-[1.5] text-white py-2 font-semibold "
            "bg-gradient-to-r from-red-900 to-orange-900"
        ))
        btn.update()


def _get_sim_params(storage: dict) -> dict:
    from backend.model_io import build_state_dict, SOLID_PRESETS
    state = build_state_dict(storage)
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from UI_components.cht_models import SolidBodySpec
        converted = []
        for b in state.get("bodies", []):
            if isinstance(b, dict):
                try:
                    converted.append(SolidBodySpec.from_dict(b))
                except Exception as _e:
                    print(f"[params] SolidBodySpec.from_dict failed for {b.get('name','?')}: {_e}")
                    converted.append(b)
            else:
                converted.append(b)
        state["bodies"] = converted
    except ImportError:
        pass
    state["project_name"]   = storage.get("project_name", "cht_result")
    state["domain_walls"]   = storage.get("face_walls",   [])
    state["domain_outlets"] = storage.get("face_outlets", ["+X"])
    return state


# ─────────────────────────────────────────────────────────────────────────────
#  Button event handlers
# ─────────────────────────────────────────────────────────────────────────────

def _on_bodies_changed(names: list[str], storage: dict, ui_state: dict):
    bcs_refs = ui_state.get("bcs_refs", {})
    update_body_dropdowns(bcs_refs, names, storage)
    pb = ui_state.get("preview_btn")
    if pb:
        if names:
            pb.props(remove="disable")
        else:
            pb.props("disable")
    bp = ui_state.get("bodies_panel_ref")
    if bp:
        bp.refresh()
    async def _auto_calc():
        bodies = storage.get("bodies", [])
        lx, ly, lz = await run.io_bound(_calc_domain_from_stls, bodies)
        if lx is None:
            return
        dom_refs = ui_state.get("domain_refs", {})
        for k, v in [("Lx", lx), ("Ly", ly), ("Lz", lz)]:
            storage[k] = v
            inp = dom_refs.get(k)
            if inp:
                inp.value = v
        upd = dom_refs.get("_update_grid_lbl")
        if upd:
            upd()
    ui.timer(0.05, _auto_calc, once=True)


def _calc_domain_from_stls(bodies: list):
    try:
        import pyvista as pv
        xmin, xmax, ymin, ymax, zmin, zmax = 1e9, -1e9, 1e9, -1e9, 1e9, -1e9
        found = False
        for b in bodies:
            path = b.get("stl_path", "") if isinstance(b, dict) else getattr(b, "stl_path", "")
            if not path or not Path(path).exists():
                continue
            try:
                mesh = pv.read(path)
                bnd = mesh.bounds
                xmin = min(xmin, bnd[0]); xmax = max(xmax, bnd[1])
                ymin = min(ymin, bnd[2]); ymax = max(ymax, bnd[3])
                zmin = min(zmin, bnd[4]); zmax = max(zmax, bnd[5])
                found = True
            except Exception:
                pass
        if not found:
            return None, None, None
        dx_mm = max((xmax - xmin) * 1.4, 10.0)
        dy_mm = max((ymax - ymin) * 1.4, 10.0)
        dz_mm = max((zmax - zmin) * 1.4, 10.0)
        return round(dx_mm / 1000, 3), round(dy_mm / 1000, 3), round(dz_mm / 1000, 3)
    except Exception:
        return None, None, None


async def _on_preview_click(storage: dict, ui_state: dict):
    bodies = storage.get("bodies", [])
    print(f"[PREVIEW] {len(bodies)} bodies in storage: {[b.get('name','?') if isinstance(b,dict) else getattr(b,'name','?') for b in bodies]}")
    if not bodies:
        ui.notify("Add at least one solid body first.", type="warning"); return

    pb = ui_state.get("preview_btn")
    if pb: pb.props("disable")
    log = ui_state.get("log_area")
    if log: log.push("⊲  Building domain preview…")
    sl = ui_state.get("status_lbl")
    if sl: sl.set_text("Building preview…")

    params = _get_sim_params(storage)
    from backend.runner import launch_preview_thread
    launch_preview_thread(params)
    ui_state["prev_log_len"] = 0
    ui_state["_prev_timer"].activate()
    ui.notify("Domain preview building in background…", type="info", timeout=3000)


async def _on_run_click(storage: dict, ui_state: dict):
    if ui_state["sim_running"]:
        abort_sim()
        log = ui_state.get("log_area")
        if log: log.push("⏹  Abort requested…")
        return

    bodies = storage.get("bodies", [])
    if not bodies or not bodies[0].get("stl_path", ""):
        ui.notify("Add at least one solid body with an STL file.", type="warning"); return

    confirmed = await _confirm_run_dialog(storage)
    if not confirmed:
        return

    ui_state["sim_running"] = True
    _set_run_btn_running(ui_state)
    reset_sim_state()

    log = ui_state.get("log_area")
    if log: log.push("▶  Simulation started…")

    params = _get_sim_params(storage)
    launch_sim_thread(params)
    ui_state["prev_log_len_sim"] = 0
    ui_state["_sim_timer"].activate()


async def _confirm_run_dialog(storage: dict) -> bool:
    dom = dict(Lx=storage.get("Lx",5), Ly=storage.get("Ly",1), Lz=storage.get("Lz",1))
    dx  = storage.get("dx_mm", 10.0)
    nx  = round(dom["Lx"]*1000/dx); ny = round(dom["Ly"]*1000/dx); nz = round(dom["Lz"]*1000/dx)
    n_b = len(storage.get("bodies", []))
    proj = storage.get("project_name","cht_result")

    with ui.dialog() as dlg, ui.card().classes(
        "bg-neutral-800 border border-neutral-600 min-w-80 gap-3"
    ):
        ui.label("Start Simulation?").classes("text-base font-bold text-slate-200")
        ui.separator().classes("bg-neutral-600")
        with ui.grid(columns=2).classes("gap-x-6 gap-y-1"):
            for label, value in [
                ("Grid",      f"{nx} × {ny} × {nz}  ({nx*ny*nz/1e6:.2f} M cells)"),
                ("Bodies",    str(n_b)),
                ("Voxel",     f"{dx:.1f} mm"),
                ("Flow",      f"{storage.get('flow_dir','+X')}  @  {storage.get('u_in',1.0):.4g} m/s"),
                ("Max iters", f"{int(storage.get('max_outer',50000)):,}"),
                ("Project",   proj),
            ]:
                ui.label(label).classes("text-xs text-slate-400")
                ui.label(value).classes("text-xs text-slate-200 font-mono")
        with ui.row().classes("justify-end gap-2 mt-2"):
            ui.button("Cancel", on_click=lambda: dlg.submit(False)).props("flat dense").classes("text-slate-400")
            ui.button("▶  Run",  on_click=lambda: dlg.submit(True)).props("unelevated dense").classes("bg-sky-700 text-white px-4")

    result = await dlg
    return bool(result)


# ─────────────────────────────────────────────────────────────────────────────
#  Save / Load model
# ─────────────────────────────────────────────────────────────────────────────

def _is_headless() -> bool:
    import platform, os as _os
    if platform.system() != "Windows":
        return not bool(_os.environ.get("DISPLAY") or _os.environ.get("WAYLAND_DISPLAY"))
    return False


async def _new_model_dialog(storage: dict, ui_state: dict):
    """Confirm then reset all storage fields to defaults."""
    from backend.state import DEFAULT_STATE
    import copy

    with ui.dialog() as dlg, ui.card().classes(
        "bg-neutral-800 border border-neutral-600 gap-3"
    ).style("min-width:360px"):
        ui.label("New Model").classes("text-sm font-bold text-slate-200")
        ui.label(
            "This will clear all bodies, settings, and results. Continue?"
        ).classes("text-xs text-slate-400")
        with ui.row().classes("justify-end gap-2 w-full mt-2"):
            ui.button("Cancel", on_click=lambda: dlg.submit(False)).props(
                "flat dense"
            ).classes("text-slate-400")
            ui.button("Clear All", on_click=lambda: dlg.submit(True)).props(
                "unelevated dense"
            ).classes("bg-red-700 text-white px-4")

    result = await dlg
    if not result:
        return

    for k, v in DEFAULT_STATE.items():
        storage[k] = copy.deepcopy(v)

    proj_inp = ui_state.get("proj_input")
    if proj_inp:
        proj_inp.value = storage["project_name"]

    # FIX: notify first so it's visible before page reload wipes everything
    ui.notify("New model created.", type="positive")
    await asyncio.sleep(0.2)
    ui.navigate.reload()


async def _save_model_dialog(storage: dict, ui_state: dict):
    from backend.runner import _PROJECTS_DIR

    default_name = storage.get("project_name", "untitled") + ".chtmdl"

    if _is_headless():
        dest = _PROJECTS_DIR / default_name
        ok = await run.io_bound(save_model_file, storage, str(dest))
        if ok:
            ui.notify(f"💾  Saved → projects/{default_name}", type="positive")
            log = ui_state.get("log_area")
            if log: log.push(f"💾  Model saved → {dest}")
        else:
            ui.notify("Failed to save model.", type="negative")
    else:
        import tkinter as _tk
        from tkinter import filedialog as _fd

        def _pick():
            root = _tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
            p = _fd.asksaveasfilename(
                title="Save FluxCore3D Model",
                initialdir=str(_PROJECTS_DIR),
                initialfile=default_name,
                defaultextension=".chtmdl",
                filetypes=[("FluxCore3D Model", "*.chtmdl"), ("All files", "*.*")],
            )
            root.destroy()
            return p or ""

        result = await run.io_bound(_pick)
        if result:
            ok = await run.io_bound(save_model_file, storage, result)
            if ok:
                ui.notify(f"💾  Saved: {Path(result).name}", type="positive")
                log = ui_state.get("log_area")
                if log: log.push(f"💾  Model saved → {result}")
            else:
                ui.notify("Failed to save model.", type="negative")


def _server_file_picker(title: str, folder: Path, extensions: list[str]):
    selected = {}
    folder.mkdir(parents=True, exist_ok=True)
    files = sorted([f for f in folder.iterdir()
                    if f.is_file() and f.suffix.lower() in extensions],
                   key=lambda f: f.stat().st_mtime, reverse=True)

    ui.label(f"📁  {folder}").classes("text-xs text-slate-500 font-mono")
    if not files:
        ui.label("No files found in this folder.").classes("text-xs text-slate-400 italic")
    else:
        sel_lbl = ui.label("No file selected").classes("text-xs text-slate-400")
        def _pick(p):
            selected["path"] = str(p)
            sel_lbl.set_text(f"✔  {p.name}")
            sel_lbl.classes(replace="text-xs text-emerald-400")
        with ui.scroll_area().classes("w-full border border-neutral-700 rounded").style("max-height:200px"):
            for f in files:
                size_kb = f.stat().st_size // 1024
                mtime   = __import__("datetime").datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                with ui.row().classes(
                    "w-full items-center gap-2 px-2 py-1 hover:bg-neutral-700 cursor-pointer rounded"
                ).on("click", lambda _, p=f: _pick(p)):
                    ui.icon("description").classes("text-slate-400 text-sm")
                    ui.label(f.name).classes("text-xs text-slate-200 flex-1")
                    ui.label(f"{size_kb} KB  {mtime}").classes("text-xs text-slate-500")
    return selected


async def _load_model_dialog(storage: dict, ui_state: dict):
    from backend.runner import _PROJECTS_DIR
    headless = _is_headless()

    with ui.dialog() as dlg, ui.card().classes(
        "bg-neutral-800 border border-neutral-600 gap-4"
    ).style("min-width:520px"):
        ui.label("Load Model").classes("text-base font-bold text-slate-200")
        loaded: dict = {}
        server_sel: dict = {}

        if headless:
            ui.label("Select a model from the server projects folder, or upload from your computer:").classes(
                "text-xs text-slate-400")
            server_sel = _server_file_picker("Projects folder", _PROJECTS_DIR, [".chtmdl", ".json"])
            ui.separator().classes("bg-neutral-700")
            ui.label("Or upload from your computer:").classes("text-xs text-slate-400")

        async def handle_upload(e):
            try:
                fname = e.file.name if hasattr(e, "file") else e.name
                raw   = await e.file.read() if hasattr(e, "file") else e.content.read()
                text  = raw.decode("utf-8")
            except Exception as err:
                status_lbl.set_text(f"Read error: {err}")
                status_lbl.classes(replace="text-xs text-red-400")
                return
            try:
                loaded["model"] = json.loads(text)
                loaded["name"]  = fname
                status_lbl.set_text(f"Ready: {fname}")
                status_lbl.classes(replace="text-xs text-emerald-400")
            except Exception as exc:
                status_lbl.set_text(f"Parse error: {exc}")
                status_lbl.classes(replace="text-xs text-red-400")

        ui.upload(
            label="Upload .chtmdl",
            auto_upload=True,
            on_upload=handle_upload,
        ).props("accept='.chtmdl,.json' flat dense").classes("w-full")
        status_lbl = ui.label("No file selected").classes("text-xs text-slate-500")

        with ui.row().classes("justify-end gap-2 mt-1"):
            ui.button("Cancel", on_click=lambda: dlg.submit(False)).props("flat dense").classes("text-slate-400")
            ui.button("Apply",  on_click=lambda: dlg.submit(True)).props("unelevated dense").classes("bg-sky-700 text-white")

    result = await dlg
    if not result: return

    if headless and server_sel.get("path"):
        m = load_model_file(server_sel["path"])
        if m:
            restore_storage_from_model(storage, m)
            storage["_load_msg"] = f"Model loaded: {Path(server_sel['path']).name}"
            ui.navigate.reload()
        else:
            ui.notify("Failed to read model file.", type="negative")
    elif "model" in loaded:
        restore_storage_from_model(storage, loaded["model"])
        storage["_load_msg"] = f"Model loaded: {loaded.get('name', '')}"
        ui.navigate.reload()


# ─────────────────────────────────────────────────────────────────────────────
#  Load Results NPZ
# ─────────────────────────────────────────────────────────────────────────────

async def _load_results_dialog(storage: dict, ui_state: dict):
    from backend.runner import _SIM_OUT_DIR
    headless = _is_headless()

    with ui.dialog() as dlg, ui.card().classes(
        "bg-neutral-800 border border-neutral-600 gap-4"
    ).style("min-width:520px"):
        ui.label("Load Results NPZ").classes("text-base font-bold text-slate-200")
        loaded: dict = {}
        server_sel: dict = {}

        if headless:
            ui.label("Select a results file from the server, or upload from your computer:").classes(
                "text-xs text-slate-400")
            all_npz = sorted(
                [f for f in _SIM_OUT_DIR.rglob("*.npz")],
                key=lambda f: f.stat().st_mtime, reverse=True
            )
            if not all_npz:
                ui.label(f"No .npz files found under {_SIM_OUT_DIR}").classes(
                    "text-xs text-slate-400 italic")
            else:
                ui.label(f"📁  {_SIM_OUT_DIR}").classes("text-xs text-slate-500 font-mono")
                sel_lbl = ui.label("No file selected").classes("text-xs text-slate-400")
                def _pick_npz(p):
                    server_sel["path"] = str(p)
                    server_sel["name"] = p.name
                    sel_lbl.set_text(f"✔  {p.name}")
                    sel_lbl.classes(replace="text-xs text-emerald-400")
                with ui.scroll_area().classes("w-full border border-neutral-700 rounded").style("max-height:200px"):
                    for f in all_npz:
                        size_mb = f.stat().st_size / 1_048_576
                        mtime   = __import__("datetime").datetime.fromtimestamp(
                            f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                        rel = f.relative_to(_SIM_OUT_DIR)
                        with ui.row().classes(
                            "w-full items-center gap-2 px-2 py-1 hover:bg-neutral-700 cursor-pointer rounded"
                        ).on("click", lambda _, p=f: _pick_npz(p)):
                            ui.icon("analytics").classes("text-sky-400 text-sm")
                            ui.label(str(rel)).classes("text-xs text-slate-200 flex-1")
                            ui.label(f"{size_mb:.1f} MB  {mtime}").classes("text-xs text-slate-500")

            ui.separator().classes("bg-neutral-700")
            ui.label("Or upload from your computer:").classes("text-xs text-slate-400")

        async def handle_upload(e):
            try:
                fname = e.file.name if hasattr(e, "file") else e.name
                raw   = await e.file.read() if hasattr(e, "file") else e.content.read()
            except Exception as err:
                status_lbl.set_text(f"Read error: {err}")
                status_lbl.classes(replace="text-xs text-red-400")
                return
            tmp = tempfile.NamedTemporaryFile(
                delete=False, suffix=".npz", dir=tempfile.gettempdir()
            )
            tmp.write(raw); tmp.close()
            loaded["path"] = tmp.name
            loaded["name"] = fname
            status_lbl.set_text(f"Ready: {fname}")
            status_lbl.classes(replace="text-xs text-emerald-400")

        ui.upload(
            label="Upload .npz from your computer",
            auto_upload=True,
            on_upload=handle_upload,
        ).props("accept='.npz' flat dense").classes("w-full")
        status_lbl = ui.label("No file selected").classes("text-xs text-slate-500")

        with ui.row().classes("justify-end gap-2 mt-1"):
            ui.button("Cancel", on_click=lambda: dlg.submit(False)).props("flat dense").classes("text-slate-400")
            ui.button("Render", on_click=lambda: dlg.submit(True)).props("unelevated dense").classes("bg-sky-700 text-white")

    result = await dlg
    if not result: return

    if server_sel.get("path"):
        await _load_npz_into_viewer(server_sel["path"], storage, ui_state,
                                     name=server_sel.get("name", ""))
    elif loaded.get("path"):
        await _load_npz_into_viewer(loaded["path"], storage, ui_state,
                                     name=loaded.get("name", ""))


async def _load_npz_into_viewer(
    npz_path: str, storage: dict, ui_state: dict, name: str = ""
):
    if ui_state.get("_results_loading"):
        return
    ui_state["_results_loading"] = True
    try:
        await _load_npz_into_viewer_impl(npz_path, storage, ui_state, name)
    finally:
        ui_state["_results_loading"] = False


def _adaptive_fmt(fmin: float, fmax: float) -> str:
    """Return a format string that gives sensible precision for the range."""
    rng = fmax - fmin
    if rng < 0.001:  return ".6f"
    if rng < 0.01:   return ".5f"
    if rng < 0.1:    return ".4f"
    if rng < 1.0:    return ".3f"
    if rng < 10.0:   return ".2f"
    return ".1f"


# Field → display units map
_FIELD_UNITS = {
    "T":     "°C",
    "speed": "m/s",
    "rho":   "kg/m³",
    "p":     "Pa",
    "u":     "m/s",
    "v":     "m/s",
    "w":     "m/s",
}


async def _load_npz_into_viewer_impl(
    npz_path: str, storage: dict, ui_state: dict, name: str = ""
):
    import numpy as np

    storage["result_npz"] = npz_path
    storage["vis_npz"]    = npz_path

    try:
        data   = np.load(npz_path, allow_pickle=True)
        names  = list(data["body_names"]) if "body_names" in data.files else []
        flow_d = str(np.asarray(data["flow_dir"]).flat[0]) if "flow_dir" in data.files else storage.get("flow_dir","+X")
        field  = storage.get("vis_field", "T")
        if field not in data.files: field = "T"
        arr    = data[field]
        F      = (arr[0] if arr.ndim==4 else arr).astype(np.float32)
        all_v  = F[np.isfinite(F)]
        fmin   = float(all_v.min()) if len(all_v) else 0.0
        fmax   = float(all_v.max()) if len(all_v) else 1.0
        storage["result_flow_dir"] = flow_d
    except Exception:
        names = []; flow_d = storage.get("flow_dir","+X"); fmin = 0.0; fmax = 1.0; field = "T"

    populate_vis_bodies(storage, names, ui_state.get("vis_refs", {}))

    sl  = ui_state.get("status_lbl")
    lbl = ui_state.get("res_lbl")
    vis_npz_lbl = ui_state.get("vis_refs", {}).get("npz_lbl")
    if vis_npz_lbl: vis_npz_lbl.set_text(name or Path(npz_path).name)
    if sl: sl.set_text("Exporting temperature surfaces…")
    if lbl: lbl.set_text(f"Exporting {name or Path(npz_path).name}…")
    ui.notify("Exporting results to GLTF…", type="info", timeout=3000)

    export_out = await run.io_bound(export_results_gltf, npz_path, storage)

    body_list = export_out.get("bodies", [])
    if not body_list and not export_out.get("streamlines"):
        ui.notify("Results export failed — check terminal.", type="negative")
        if sl: sl.set_text("Export failed.")
        return

    fmin  = export_out.get("fmin", fmin)
    fmax  = export_out.get("fmax", fmax)
    field = export_out.get("field", field)

    import matplotlib
    cmap = matplotlib.colormaps.get_cmap(export_out.get("cmap", "inferno"))

    # FIX: adaptive precision for legend mean temperature labels
    _fmt = _adaptive_fmt(fmin, fmax)
    _units = _FIELD_UNITS.get(field, "")

    legend = []
    for bm in body_list:
        if bm.get("t_min") is not None and bm.get("name") != "Fluid domain":
            mean_t = (bm["t_min"] + bm["t_max"]) / 2.0
            mc     = cmap(float(max(0, min(1, (mean_t - fmin) / max(fmax - fmin, 1e-6)))))
            hx     = "#{:02x}{:02x}{:02x}".format(int(mc[0]*255),int(mc[1]*255),int(mc[2]*255))
            # Use correct unit suffix in legend
            unit_lbl = "°C" if field == "T" else _units
            legend.append(dict(name=f"{bm['name']} {format(mean_t, _fmt)} {unit_lbl}".strip(), color=hx))

    # FIX: build colorbar ticks with adaptive precision + correct field units
    cbar_ticks = []
    for i in range(6):
        t  = fmin + i * (fmax - fmin) / 5.0
        mc = cmap(i / 5.0)
        hx = "#{:02x}{:02x}{:02x}".format(int(mc[0]*255), int(mc[1]*255), int(mc[2]*255))
        cbar_ticks.append(dict(label=format(t, _fmt), color=hx))

    ui_state["result_cbar"] = dict(
        ticks=cbar_ticks,
        field=field,
        units=_units,   # passed to overlay label
        fmin=fmin,
        fmax=fmax,
        cmap=export_out.get("cmap", "inferno"),
    )
    ui_state["result_grid_size"] = export_out.get("grid_size", [100, 100, 100])

    all_gltf = list(body_list)
    if export_out.get("streamlines"):
        all_gltf.append(dict(name="Streamlines",
                              gltf_url=export_out["streamlines"],
                              is_lines=True))

    import time as _time
    _ts = int(_time.time())
    for _item in all_gltf:
        if "gltf_url" in _item:
            base = _item["gltf_url"].split("?")[0]
            _item["gltf_url"] = f"{base}?v={_ts}"

    ui_state["result_meta"]   = all_gltf
    ui_state["result_legend"] = legend

    _show_viewer_mode(ui_state, "results")

    # FIX: adaptive format in status label
    if lbl: lbl.set_text(f"{name or Path(npz_path).name}  {field}: {format(fmin, _fmt)}–{format(fmax, _fmt)} {_units}".strip())
    if sl: sl.set_text("Results loaded.")
    fb = ui_state.get("full_viewer_btn")
    if fb: fb.props(remove="disable")


# ─────────────────────────────────────────────────────────────────────────────
#  Visualize / full viewer
# ─────────────────────────────────────────────────────────────────────────────

async def _browse_npz_dialog(storage: dict, ui_state: dict):
    await _load_results_dialog(storage, ui_state)


async def _launch_viz(storage: dict, ui_state: dict):
    npz = storage.get("vis_npz", "") or storage.get("result_npz", "")
    if not npz:
        ui.notify("Load a results NPZ first.", type="warning"); return

    log = ui_state.get("log_area")
    if log: log.push(f"📊  Rendering {Path(npz).name}  field={storage.get('vis_field','T')}")
    sl = ui_state.get("status_lbl")
    if sl: sl.set_text("Rendering plot…")

    await _load_npz_into_viewer(npz, storage, ui_state)


def _launch_full_viewer(storage: dict):
    npz = storage.get("result_npz", "")
    if not npz:
        ui.notify("No results loaded.", type="warning"); return

    def _open():
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from UI_components.cht_sim_imports import LBMCHT3D_Torch, HAS_SIM
            if not HAS_SIM:
                return
            LBMCHT3D_Torch.view_snapshots_pyvista_3d_single_npz(
                npz_path=npz,
                fields=["T","speed","u","v","w","rho"],
                initial_field="T",
                seed_mode="inlet_plane",
                use_global_clim=True,
                flow_dir=storage.get("result_flow_dir", "+X"),
                cmap="inferno",
            )
        except Exception as exc:
            print(f"[FullViewer] {exc}")

    import threading
    threading.Thread(target=_open, daemon=True).start()
    ui.notify("Full viewer launching in a separate window…", type="info")


# ─────────────────────────────────────────────────────────────────────────────
#  Thermal diagnostics dialog
# ─────────────────────────────────────────────────────────────────────────────

def _show_diagnostics(npz_path: str, ui_state: dict):
    import numpy as np

    try:
        data = np.load(npz_path, allow_pickle=True)
        if "T" not in data.files:
            return
        arr  = data["T"]
        T    = (arr[0] if arr.ndim == 4 else arr).astype(float)
        body_names = list(data["body_names"]) if "body_names" in data.files else []
        rows = []
        solid_union = None

        for name in body_names:
            key = f"solid_{name}"
            if key not in data.files:
                rows.append(dict(name=name, tmin="—", tmean="—", tmax="—", nvox="—"))
                continue
            m = data[key].astype(bool)
            if m.ndim == 4: m = m[0]
            solid_union = m if solid_union is None else (solid_union | m)
            if not m.any():
                rows.append(dict(name=name, tmin="—", tmean="—", tmax="—", nvox="—"))
                continue
            Tb = T[m]
            rows.append(dict(
                name=name,
                tmin=f"{Tb.min():.2f}", tmean=f"{Tb.mean():.2f}",
                tmax=f"{Tb.max():.2f}", nvox=f"{int(m.sum()):,}",
            ))

        if solid_union is not None:
            Ts = T[solid_union]; Tf = T[~solid_union]
            rows.append(dict(name="ALL SOLID",
                tmin=f"{Ts.min():.2f}", tmean=f"{Ts.mean():.2f}",
                tmax=f"{Ts.max():.2f}", nvox=f"{int(solid_union.sum()):,}",
                summary=True, col="amber"))
            rows.append(dict(name="ALL FLUID",
                tmin=f"{Tf.min():.2f}", tmean=f"{Tf.mean():.2f}",
                tmax=f"{Tf.max():.2f}", nvox=f"{int((~solid_union).sum()):,}",
                summary=True, col="sky"))
    except Exception as exc:
        print(f"[Diagnostics] {exc}")
        return

    async def _show():
        with ui.dialog() as dlg, ui.card().classes(
            "bg-neutral-800 border border-neutral-600 min-w-xl gap-3"
        ):
            ui.label("Thermal Diagnostics — Post-Solve Summary").classes(
                "text-base font-bold text-slate-200"
            )
            html_rows = ""
            for row in rows:
                is_sum = row.get("summary", False)
                col    = row.get("col", "")
                style  = (
                    f"color:#{'FFD580' if col=='amber' else '80D4FF'};font-weight:bold;"
                    if is_sum else ""
                )
                html_rows += (
                    f"<tr>"
                    f"<td style='{style}'>{row['name']}</td>"
                    f"<td style='{style};text-align:right'>{row['tmin']}</td>"
                    f"<td style='{style};text-align:right'>{row['tmean']}</td>"
                    f"<td style='{style};text-align:right'>{row['tmax']}</td>"
                    f"<td style='{style};text-align:right'>{row['nvox']}</td>"
                    f"</tr>"
                )
            ui.html(f"""
                <table class="diag-table w-full" style="border-collapse:collapse">
                    <thead>
                        <tr>
                            <th style="text-align:left">Body</th>
                            <th>T min (°C)</th>
                            <th>T mean (°C)</th>
                            <th>T max (°C)</th>
                            <th>N voxels</th>
                        </tr>
                    </thead>
                    <tbody>{html_rows}</tbody>
                </table>
            """)
            with ui.row().classes("justify-end mt-2"):
                ui.button("Close", on_click=dlg.close).props("flat dense").classes("text-slate-400")
        dlg.open()

    ui.timer(0.1, _show, once=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ in {"__main__", "__mp_main__"}:
    # FIX: raise Starlette's per-part upload limit so large STL files don't stall at 99%
    try:
        from starlette.formparsers import MultiPartParser
        MultiPartParser.max_part_size = 100 * 1024 * 1024  # 100 MB
    except Exception:
        pass

    # FIX: kill Python process 5 s after the last browser tab closes
    _disconnect_tasks: list = []

    async def _schedule_shutdown():
        await asyncio.sleep(5)
        import signal
        os.kill(os.getpid(), signal.SIGTERM)

    async def _on_disconnect():
        if _disconnect_tasks:
            t = _disconnect_tasks.pop()
            if not t.done(): t.cancel()
        _disconnect_tasks.append(asyncio.ensure_future(_schedule_shutdown()))

    async def _on_connect():
        if _disconnect_tasks:
            t = _disconnect_tasks.pop()
            if not t.done(): t.cancel()

    app.on_disconnect(_on_disconnect)
    app.on_connect(_on_connect)

    ui.run(
        title="FluxCore3D",
        dark=True,
        port=8080,
        reload=False,
        storage_secret="fluxcore3d-secret-2024",
        favicon="⚡",
        reconnect_timeout=30,  # FIX: default is 3s — too short for tab switches
    )
