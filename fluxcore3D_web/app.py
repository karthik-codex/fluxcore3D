"""
app.py — FluxCore3D NiceGUI web app entry point.

Layout: left control panel (tabs) + right viewer panel.

NiceGUI rules applied throughout:
  - on_change=lambda e: handler(e.value)
  - No widget references stored in app.storage.user
  - .props(remove="disable") to re-enable, NOT .props("enable")
  - Toggle buttons with Python state for radio-style exclusivity
  - await run.io_bound(...) for blocking sim work
  - ui.notify only from async context / main thread
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

# ── Path bootstrap: add parent dir so UI_components can be found ──────────────
_APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_APP_DIR))
sys.path.insert(0, str(_APP_DIR.parent))   # for UI_components sibling package

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

# Serve static CSS
app.add_static_files("/static", str(_APP_DIR / "static"))


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE
# ─────────────────────────────────────────────────────────────────────────────

@ui.page("/")
async def main_page():
    storage = get_storage()

    # ── Show deferred toast from load-model page reload ────────────────────────
    _msg = storage.pop("_load_msg", None)
    if _msg:
        ui.timer(0.3, lambda: ui.notify(f"📂  {_msg}", type="positive"), once=True)

    # ── Surface CUDA/sim status in log on page load ───────────────────────────
    # runner.py imports sim modules at module level so CUDA compiles at startup.
    def _show_sim_status():
        log = ui_state.get("log_area")
        if log is None: return
        if _RUNNER_HAS_SIM:
            log.push("✅  Sim modules loaded — CUDA extensions ready.")
        else:
            log.push(f"⚠️  Sim modules unavailable: {_RUNNER_SIM_ERR}")
    ui.timer(0.6, _show_sim_status, once=True)

    # ── Restore results if NPZ still exists from last session ─────────────────
    async def _restore_results_on_load():
        npz = storage.get("result_npz", "")
        if npz and Path(npz).exists():
            log = ui_state.get("log_area")
            if log: log.push(f"📂  Restoring last results: {Path(npz).name}")
            await _load_npz_into_viewer(npz, storage, ui_state)
    ui.timer(1.0, _restore_results_on_load, once=True)

    # ── Global UI state (per-page, not serialised to storage) ─────────────────
    ui_state = {
        "sim_running":     False,
        "prev_log_len":    0,
        "prev_log_len_sim":0,
        "image_panel":     None,   # ui.image ref for viewer
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

    # ── Custom CSS + fonts ────────────────────────────────────────────────────
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

    # ── Page title / header ───────────────────────────────────────────────────
    with ui.row().classes(
        "w-full items-center gap-3 px-4 py-0 border-b border-neutral-700"
    ).style("background:#1e1e1e"):
        ui.html('<svg width="356" height="72" viewBox="0 0 178 36" fill="none" xmlns="http://www.w3.org/2000/svg"><defs><linearGradient id="fc3d_i" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stop-color="#000004"/><stop offset="13%" stop-color="#1B0C41"/><stop offset="26%" stop-color="#4A0C6B"/><stop offset="38%" stop-color="#781C6D"/><stop offset="50%" stop-color="#A52C60"/><stop offset="62%" stop-color="#CF4446"/><stop offset="74%" stop-color="#ED6925"/><stop offset="86%" stop-color="#FB9A06"/><stop offset="93%" stop-color="#F7D13D"/><stop offset="100%" stop-color="#FCFFA4"/></linearGradient><linearGradient id="fc3d_bg" x1="0" y1="1" x2="1" y2="0"><stop offset="0%" stop-color="#0072FF"/><stop offset="100%" stop-color="#00C6FF"/></linearGradient><linearGradient id="fc3d_wm" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stop-color="#00C6FF"/><stop offset="100%" stop-color="#00FFA3"/></linearGradient><filter id="fc3d_glow"><feGaussianBlur stdDeviation="1.1" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter><clipPath id="fc3d_cc"><rect x="5" y="5" width="24" height="24" rx="1.5"/></clipPath></defs><rect x="5" y="5" width="24" height="24" rx="1.5" fill="#080c14" stroke="#1a2440" stroke-width="1"/><rect x="5" y="5" width="24" height="24" rx="1.5" fill="url(#fc3d_bg)" opacity="0.08" clip-path="url(#fc3d_cc)"/><path d="M0,14 C5,14 7,11 11,11 S17,14 22,14 S27,17 34,17" stroke="url(#fc3d_i)" stroke-width="1.3" fill="none" filter="url(#fc3d_glow)" opacity="1"/><path d="M0,18 C5,18 7,15 11,15 S17,18 22,18 S27,21 34,21" stroke="url(#fc3d_i)" stroke-width="0.9" fill="none" opacity="0.65"/><path d="M0,22 C5,22 7,19 11,19 S17,22 22,22 S27,25 34,25" stroke="url(#fc3d_i)" stroke-width="0.6" fill="none" opacity="0.35"/><line x1="11" y1="5" x2="11" y2="2" stroke="#1e3a4a" stroke-width="1.3" stroke-linecap="round"/><line x1="17" y1="5" x2="17" y2="2" stroke="#1e3a4a" stroke-width="1.3" stroke-linecap="round"/><line x1="23" y1="5" x2="23" y2="2" stroke="#1e3a4a" stroke-width="1.3" stroke-linecap="round"/><line x1="11" y1="29" x2="11" y2="32" stroke="#1e3a4a" stroke-width="1.3" stroke-linecap="round"/><line x1="17" y1="29" x2="17" y2="32" stroke="#1e3a4a" stroke-width="1.3" stroke-linecap="round"/><line x1="23" y1="29" x2="23" y2="32" stroke="#1e3a4a" stroke-width="1.3" stroke-linecap="round"/><line x1="5" y1="12" x2="2" y2="12" stroke="#1e3a4a" stroke-width="1.3" stroke-linecap="round"/><line x1="5" y1="18" x2="2" y2="18" stroke="#1e3a4a" stroke-width="1.3" stroke-linecap="round"/><line x1="5" y1="24" x2="2" y2="24" stroke="#1e3a4a" stroke-width="1.3" stroke-linecap="round"/><line x1="29" y1="12" x2="32" y2="12" stroke="#1e3a4a" stroke-width="1.3" stroke-linecap="round"/><line x1="29" y1="18" x2="32" y2="18" stroke="#1e3a4a" stroke-width="1.3" stroke-linecap="round"/><line x1="29" y1="24" x2="32" y2="24" stroke="#1e3a4a" stroke-width="1.3" stroke-linecap="round"/><text x="40" y="15" font-family="Courier New,monospace" font-size="10" font-weight="700" letter-spacing="3" fill="url(#fc3d_wm)">FLUX</text><text x="40" y="28" font-family="Courier New,monospace" font-size="10" font-weight="700" letter-spacing="3" fill="#e2e8f0">CORE<tspan fill="#00FFA3" letter-spacing="1.5">3D</tspan></text></svg>').classes("flex-shrink-0")
        ui.label("Thermal Intelligence for Electronics Cooling").classes(
            "text-large text-slate-500 tracking-widest font-mono"
        )
        ui.space()
        # ── Project + model buttons (top-right) ───────────────────────────────
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

    # ── Main body: splitter ───────────────────────────────────────────────────
    with ui.splitter(value=25).classes("w-full flex-1").style(
        "height: calc(100vh - 52px)"
    ) as splitter:
        with splitter.before:
            _build_control_panel(storage, ui_state)
        with splitter.after:
            _build_viewer_panel(storage, ui_state)

    # ── Polling timer: sim progress + preview done ────────────────────────────
    async def _poll_sim():
        try:
            if not _sim_state["running"] and not _sim_state["done"]:
                return

            # Drain log messages
            log: ui.log = ui_state.get("log_area")
            prev_len = ui_state["prev_log_len_sim"]
            msgs = _sim_state["log"]
            for msg in msgs[prev_len:]:
                if log:
                    log.push(msg)
            ui_state["prev_log_len_sim"] = len(msgs)

            # Progress bar
            pb = ui_state.get("progress_bar")
            if pb:
                pb.value = _sim_state["pct"] / 100.0

            # Status label
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
                    # Clear done BEFORE the await so re-entrant timer ticks skip this block
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

            # Drain log
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
                    # Cache-bust preview GTLFs so browser re-fetches overwritten files
                    import time as _time
                    _ts = int(_time.time())
                    for _bm in (_preview_state.get("body_meta") or []):
                        if "gltf_url" in _bm:
                            base = _bm["gltf_url"].split("?")[0]
                            _bm["gltf_url"] = f"{base}?v={_ts}"
                    # Only refresh if still in domain mode — never clobber results view
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

    # Stash timers in ui_state so handlers can activate them
    ui_state["_sim_timer"]  = _sim_timer
    ui_state["_prev_timer"] = _prev_timer

    # ── Wire global button callbacks ──────────────────────────────────────────
    ui_state["_storage"]    = storage
    ui_state["_sim_timer"]  = _sim_timer
    ui_state["_prev_timer"] = _prev_timer


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

            # ── Domains tab ───────────────────────────────────────────────
            with ui.tab_panel(t_dom).classes("gap-3 flex flex-col"):
                bp, dom_refs = build_domains_tab(
                    storage,
                    on_bodies_changed=lambda names: _on_bodies_changed(names, storage, ui_state),
                    on_save_model=lambda: _save_model_dialog(storage, ui_state),
                    on_load_model=lambda: _load_model_dialog(storage, ui_state),
                )
                ui_state["bodies_panel_ref"] = bp
                ui_state["domain_refs"] = dom_refs

            # ── BCs tab ───────────────────────────────────────────────────
            with ui.tab_panel(t_bc).classes("gap-3 flex flex-col"):
                face_grid, bcs_refs = build_bcs_tab(
                    storage,
                    body_names=[b.get("name","") for b in storage.get("bodies",[])],
                )
                ui_state["face_grid_ref"] = face_grid
                ui_state["bcs_refs"]      = bcs_refs

            # ── Solver tab ────────────────────────────────────────────────
            with ui.tab_panel(t_sol).classes("gap-3 flex flex-col"):
                build_solver_tab(storage)

            # ── Visualize tab ─────────────────────────────────────────────
            with ui.tab_panel(t_vis).classes("gap-3 flex flex-col"):
                vis_refs = build_visualize_tab(
                    storage,
                    on_render=lambda: _launch_viz(storage, ui_state),
                    on_browse_npz=lambda: _browse_npz_dialog(storage, ui_state),
                )
                ui_state["vis_refs"] = vis_refs

        # ── Status / progress / buttons ───────────────────────────────────────
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
    # Stop preview timer from clobbering results mode with stale refreshes
    if mode == "results":
        pt = ui_state.get("_prev_timer")
        if pt: pt.deactivate()
    r = ui_state.get("scene_refresh")
    if r: r()


def _set_viewer_image(b64: str, ui_state: dict, mode: str = "domain"):
    key = "preview_b64" if mode=="domain" else "results_b64"
    ui_state[key] = b64


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

        # ── Snapshot button (right side of info bar) ─────────────────────────
        with ui.row().classes("w-full items-center px-3 pt-1 gap-2").style("flex-shrink:0"):
            info_lbl = ui.label("Add bodies → click Preview to load here.").classes(
                "text-xs text-slate-500 flex-1")
            ui_state["info_lbl"] = info_lbl
            res_lbl = ui.label("").classes("text-xs text-slate-500 flex-1")
            res_lbl.set_visibility(False)
            ui_state["res_lbl"] = res_lbl

            async def _snapshot():
                """Capture the Three.js canvas and trigger browser download."""
                mode = ui_state.get("viewer_mode", "domain")
                fname = f"fluxcore3d_{'preview' if mode=='domain' else 'results'}.png"
                await ui.run_javascript(f"""
                    (function() {{
                        const canvases = document.querySelectorAll('canvas');
                        if (!canvases.length) {{ alert('No canvas found'); return; }}
                        // Pick the largest canvas (the Three.js scene)
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
            res_meta  = ui_state.get("result_meta") or []   # [{name,gltf_url,t_min,t_max}]

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
                            # Auto-fit camera to domain bounds
                            _rg = ui_state.get("result_grid_size") or [100, 100, 100]
                            rx, ry, rz = _rg
                            sc.move_camera(
                                x=rx*0.5, y=-ry*2.0, z=rz*1.5,
                                look_at_x=rx*0.5, look_at_y=ry*0.5, look_at_z=rz*0.3,
                            )

                # ── Overlays (legend + colorbar) ──────────────────────────────
                legend  = body_meta if mode == "domain" else (ui_state.get("result_legend") or [])
                cbar    = ui_state.get("result_cbar") if mode == "results" else None

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
                    # Vertical colorbar on the left edge
                    ticks = cbar.get("ticks", [])
                    with ui.column().classes(
                        "absolute bottom-6 left-3 gap-0"
                    ).style("pointer-events:none"):
                        ui.label(f"{cbar.get('field','T')} [°C]").classes(
                            "text-xs font-bold text-slate-300 mb-1")
                        with ui.row().classes("gap-1 items-stretch"):
                            # Gradient bar
                            grad_colors = ",".join(t["color"] for t in ticks)
                            ui.element("div").style(
                                f"width:14px; height:120px; border-radius:3px;"
                                f"background:linear-gradient(to top, {grad_colors});"
                                f"flex-shrink:0"
                            )
                            # Tick labels
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

        # ── Terminal output panel ──────────────────────────────────────────────
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

        # Poll sim/preview log state → push new lines to terminal log
        def _poll_stdout():
            tlog = ui_state.get("term_log")
            if tlog is None: return
            # Preview: full output in terminal
            prev_len = ui_state.get("_term_prev_len", 0)
            prev_msgs = _preview_state.get("log", [])
            for line in prev_msgs[prev_len:]:
                tlog.push(f"[PREVIEW] {line}")
            ui_state["_term_prev_len"] = len(prev_msgs)
            # Sim: read term_log (all stdout) not log (high-level only)
            sim_len = ui_state.get("_term_sim_len", 0)
            sim_msgs = _sim_state.get("term_log", [])
            for line in sim_msgs[sim_len:]:
                tlog.push(line)
            ui_state["_term_sim_len"] = len(sim_msgs)

        ui.timer(0.3, _poll_stdout)


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
#  Embedded scene populators
# ─────────────────────────────────────────────────────────────────────────────

# CSS colour name → hex (small lookup for common body colours)
_COLOUR_HEX = {
    "steelblue":"#4682B4","darkorange":"#FF8C00","mediumseagreen":"#3CB371",
    "mediumpurple":"#9370DB","tomato":"#FF6347","gold":"#FFD700",
    "hotpink":"#FF69B4","limegreen":"#32CD32","deepskyblue":"#00BFFF",
    "coral":"#FF7F50","sienna":"#A0522D",
}

def _css_to_hex(c: str) -> str:
    return _COLOUR_HEX.get(c, c) if not c.startswith("#") else c


def _populate_domain_scene(ui_state: dict, preview_state: dict):
    """
    Rebuild scene.objects Python-side then push full state to browser via init_objects.
    This is the ONLY reliable way to update a ui.scene from a timer callback.
    """
    scene = ui_state.get("domain_scene")
    if scene is None:
        return

    body_meta  = preview_state.get("body_meta", [])
    grid_cells = preview_state.get("grid_cells", [1, 1, 1])
    dx         = preview_state.get("dx_mm", 1.0)
    flow_d     = preview_state.get("flow_dir", "+X")
    nx, ny, nz = grid_cells
    W, D, H    = nx * dx, ny * dx, nz * dx

    print(f"[SCENE] init_objects push: {len(body_meta)} bodies")

    # Step 1: wipe Python-side state without sending anything to browser
    scene.objects.clear()

    # Step 2: rebuild all objects Python-side using with scene:
    # (Object3D.__init__ adds to scene.objects AND calls _create which sends
    #  run_method('create',...) — but we'll override that with init_objects below)
    # Temporarily monkeypatch _create to be a no-op so we only send one batched push
    from nicegui.elements.scene.scene_object3d import Object3D
    _orig_create = Object3D._create
    Object3D._create = lambda self: None   # suppress individual creates

    try:
        with scene:
            scene.spot_light(color="#ffffff", intensity=0.9,
                             distance=0, angle=1.2, penumbra=0.1, decay=0.5
                             ).move(W*0.5, D*2.0, H*2.0)
            _draw_box_wireframe(scene, W, D, H)
            _draw_flow_arrow(scene, W, D, H, flow_d)
            for bm in body_meta:
                pts = bm.get("points")
                if pts:
                    hex_col = _css_to_hex(bm.get("color", "steelblue"))
                    r = int(hex_col[1:3], 16) / 255.0
                    g = int(hex_col[3:5], 16) / 255.0
                    b = int(hex_col[5:7], 16) / 255.0
                    colors = [[r, g, b]] * len(pts)
                    scene.point_cloud(pts, colors, point_size=2.5)
                    print(f"  {bm['name']}: {len(pts)} pts")
            scene.axes_helper(length=max(W, D, H) * 0.15)
    finally:
        Object3D._create = _orig_create   # restore

    # Step 3: push ALL objects to browser in one batched call
    scene.run_method('init_objects', [obj.data for obj in scene.objects.values()])

    # Step 4: reposition camera
    scene.move_camera(
        x=W*0.5, y=-D*1.8, z=H*1.4,
        look_at_x=W*0.5, look_at_y=D*0.5, look_at_z=H*0.5,
    )

    info = ui_state.get("info_lbl")
    if info:
        info.set_text(
            f"Domain {W:.0f}×{D:.0f}×{H:.0f} mm  |  {dx:.1f} mm/vox  |  "
            f"flow {flow_d}  |  {len(body_meta)} bodies"
        )


def _export_results_gltf(storage: dict, npz_path: str) -> list:
    """
    Export per-body temperature surfaces as GLTF files with per-vertex RGB colour.
    Returns list of dicts: {url, name, t_min, t_max}
    Runs in a background thread (run.io_bound).
    """
    import numpy as np, traceback
    try:
        import pyvista as pv
        import matplotlib
        data      = np.load(npz_path, allow_pickle=True)
        field     = storage.get("vis_field", "T")
        if field not in data.files:
            field = "T"
        cmap_name = storage.get("vis_cmap", "inferno")

        arr = data[field]
        F   = (arr[0] if arr.ndim == 4 else arr).astype(np.float32)
        nx, ny, nz = F.shape

        all_v = F[np.isfinite(F)]
        fmin  = float(all_v.min()) if len(all_v) else 0.0
        fmax  = float(all_v.max()) if len(all_v) else 1.0
        if fmin == fmax: fmax += 1.0

        cmap = matplotlib.colormaps.get_cmap(cmap_name)

        here    = Path(__file__).parent
        res_dir = here / "static" / "results"
        res_dir.mkdir(parents=True, exist_ok=True)
        for f in res_dir.glob("*.gltf"):
            f.unlink(missing_ok=True)

        body_names   = list(data["body_names"]) if "body_names" in data.files else []
        show_fluid   = storage.get("vis_fluid", False)
        vis_bodies   = storage.get("vis_bodies", {})

        results = []
        for name in body_names:
            if vis_bodies and name in vis_bodies and not vis_bodies[name]:
                continue
            key = f"solid_{name}"
            if key not in data.files:
                continue
            mask = data[key].astype(bool)
            if mask.ndim == 4: mask = mask[0]

            try:
                # Build surface and sample temperature at each vertex
                g = pv.ImageData(dimensions=(nx,ny,nz), spacing=(1,1,1), origin=(0,0,0))
                g.point_data["mask"] = mask.reshape(-1, order="F").astype(np.uint8)
                g.point_data["T"]    = F.reshape(-1, order="F").astype(np.float32)

                surf = g.contour([0.5], scalars="mask").extract_surface(algorithm="dataset_surface")
                if surf.n_points == 0:
                    continue
                # Sample temperature from volume onto surface points
                surf = surf.sample(g)
                surf = surf.smooth(n_iter=15)

                t_vals = surf.point_data.get("T", None)
                if t_vals is None:
                    t_vals = np.full(surf.n_points, (fmin+fmax)/2, dtype=np.float32)

                # Map T → uint8 RGB via colormap
                t_norm = np.clip((t_vals - fmin) / (fmax - fmin), 0, 1)
                rgba   = cmap(t_norm)                        # (N,4) float 0-1
                rgb    = (rgba[:, :3] * 255).astype(np.uint8)
                surf.point_data["RGB"] = rgb

                safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
                out_path = res_dir / f"{safe}.gltf"
                surf.save(str(out_path))

                results.append(dict(
                    url   = f"/static/results/{safe}.gltf",
                    name  = name,
                    t_min = float(t_vals.min()),
                    t_max = float(t_vals.max()),
                ))
                print(f"[Results] {name}: {surf.n_points} pts  T={t_vals.min():.1f}–{t_vals.max():.1f}")
            except Exception as e:
                print(f"[Results] GLTF export failed for {name}: {traceback.format_exc()[-400:]}")

        # Optionally export fluid domain as transparent surface
        if show_fluid:
            try:
                fluid_mask = ~np.zeros((nx,ny,nz), dtype=bool)
                for n2 in body_names:
                    k2 = f"solid_{n2}"
                    if k2 in data.files:
                        m2 = data[k2].astype(bool)
                        if m2.ndim == 4: m2 = m2[0]
                        fluid_mask &= ~m2
                g2 = pv.ImageData(dimensions=(nx,ny,nz), spacing=(1,1,1), origin=(0,0,0))
                g2.point_data["mask"] = fluid_mask.reshape(-1,order="F").astype(np.uint8)
                g2.point_data["T"]    = F.reshape(-1,order="F").astype(np.float32)
                # Just show domain boundary faces
                surf2 = pv.Box(bounds=(0,nx,0,ny,0,nz))
                surf2 = surf2.sample(g2)
                t2 = surf2.point_data.get("T", np.full(surf2.n_points,(fmin+fmax)/2))
                t2n = np.clip((t2 - fmin)/(fmax-fmin),0,1)
                rgba2 = cmap(t2n)
                surf2.point_data["RGB"] = (rgba2[:,:3]*255).astype(np.uint8)
                surf2.save(str(res_dir / "fluid_domain.gltf"))
                opacity = storage.get("vis_fluid_opacity", 0.3)
                results.append(dict(url="/static/results/fluid_domain.gltf",
                                    name="Fluid domain", opacity=opacity,
                                    t_min=fmin, t_max=fmax))
            except: pass

        return results

    except Exception as exc:
        import traceback
        print(f"[Results export] FAILED: {traceback.format_exc()}")
        return []


async def _populate_results_scene(ui_state: dict, storage: dict, npz_path: str):
    """Export GLTF and populate the embedded results ui.scene. Async-safe."""
    scene = ui_state.get("results_scene")
    if scene is None:
        return

    ph = ui_state.get("results_scene_placeholder")
    if ph: ph.set_visibility(False)

    sl = ui_state.get("status_lbl")
    if sl: sl.set_text("Exporting results surfaces…")

    # Run heavy PyVista work off the main thread
    from nicegui import run as ngrun
    body_list = await ngrun.io_bound(_export_results_gltf, storage, npz_path)

    if not body_list:
        ui.notify("Results export failed — check terminal.", type="negative")
        if ph: ph.set_visibility(True)
        return

    data = np.load(npz_path, allow_pickle=True)
    field = storage.get("vis_field", "T")
    if field not in data.files: field = "T"
    arr = data[field]
    F   = (arr[0] if arr.ndim == 4 else arr).astype(np.float32)
    all_v = F[np.isfinite(F)]
    fmin = float(all_v.min()) if len(all_v) else 0.0
    fmax = float(all_v.max()) if len(all_v) else 1.0
    nx, ny, nz = F.shape

    scene.delete_objects()  # skip super().clear() which breaks run_method context
    with scene:
        scene.spot_light(color="#ffffff", intensity=0.7,
                         distance=0, angle=1.0, penumbra=0.2, decay=0.5
                         ).move(nx*0.5, -ny*1.6, nz*1.6)
        _draw_box_wireframe(scene, nx, ny, nz)
        for bm in body_list:
            obj = scene.gltf(bm["url"])
            if bm.get("opacity") is not None:
                obj.material(color=None, opacity=bm["opacity"], side="both")
        scene.axes_helper(length=max(nx,ny,nz)*0.15)

    scene.move_camera(
        x=nx*0.5, y=-ny*1.6, z=nz*1.2,
        look_at_x=nx*0.5, look_at_y=ny*0.5, look_at_z=nz*0.5,
    )

    res_lbl = ui_state.get("res_lbl")
    if res_lbl:
        res_lbl.set_text(
            f"{Path(npz_path).name}  |  {field}  |  "
            f"{fmin:.1f}–{fmax:.1f}  |  {len(body_list)} surfaces"
        )
    fb = ui_state.get("full_viewer_btn")
    if fb: fb.props(remove="disable")
    if sl: sl.set_text("Results loaded.")


def _draw_box_wireframe(scene, W, D, H):
    """Draw 12 edges of a bounding box in the scene."""
    corners = [
        ([0,0,0],[W,0,0]), ([0,D,0],[W,D,0]), ([0,0,H],[W,0,H]), ([0,D,H],[W,D,H]),
        ([0,0,0],[0,D,0]), ([W,0,0],[W,D,0]), ([0,0,H],[0,D,H]), ([W,0,H],[W,D,H]),
        ([0,0,0],[0,0,H]), ([W,0,0],[W,0,H]), ([0,D,0],[0,D,H]), ([W,D,0],[W,D,H]),
    ]
    for s, e in corners:
        scene.line(s, e).material(color="#2A3A5A")


def _draw_flow_arrow(scene, W, D, H, flow_d):
    """Draw a flow direction arrow in the scene."""
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


def _set_viewer_image(src: str, ui_state: dict):
    img = ui_state.get("image_panel")
    ph  = ui_state.get("viewer_placeholder")
    if img:
        img.set_source(src)
        img.style("display:block")
    if ph:
        ph.set_visibility(False)
    fb = ui_state.get("full_viewer_btn")
    if fb:
        fb.props(remove="disable")


def _set_run_btn_idle(ui_state: dict):
    btn = ui_state.get("run_btn")
    if btn:
        btn.icon  = "play_arrow"
        btn._props["label"] = "Run Model"
        btn.classes(
            replace=(
                "flex-[1.5] text-white py-2 font-semibold "
                "bg-gradient-to-r from-blue-900 to-emerald-900"
            )
        )
        btn.update()


def _set_run_btn_running(ui_state: dict):
    btn = ui_state.get("run_btn")
    if btn:
        btn.icon = "stop"
        btn._props["label"] = "Stop"
        btn.classes(
            replace=(
                "flex-[1.5] text-white py-2 font-semibold "
                "bg-gradient-to-r from-red-900 to-orange-900"
            )
        )
        btn.update()


def _get_sim_params(storage: dict) -> dict:
    """Assemble the params dict that run_simulation_blocking expects."""
    from backend.model_io import build_state_dict
    from backend.model_io import SOLID_PRESETS

    state = build_state_dict(storage)
    # Convert body dicts → SolidBodySpec objects
    # NOTE: runner._blocking_preview now uses _as_dict() which handles both plain
    # dicts and SolidBodySpec objects, so conversion here is only needed for the
    # actual simulation run. Keep as dicts for preview to avoid from_dict failures.
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from UI_components.cht_models import SolidBodySpec
        converted = []
        for b in state.get("bodies", []):
            if isinstance(b, dict):
                try:
                    converted.append(SolidBodySpec.from_dict(b))
                except Exception as _e:
                    print(f"[params] SolidBodySpec.from_dict failed for {b.get('name','?')}: {_e} — keeping dict")
                    converted.append(b)
            else:
                converted.append(b)
        state["bodies"] = converted
    except ImportError:
        pass  # fallback — runner will handle dicts

    state["project_name"] = storage.get("project_name", "cht_result")
    state["domain_walls"]   = storage.get("face_walls",   [])
    state["domain_outlets"] = storage.get("face_outlets", ["+X"])
    return state


# ─────────────────────────────────────────────────────────────────────────────
#  Button event handlers
# ─────────────────────────────────────────────────────────────────────────────

def _on_bodies_changed(names: list[str], storage: dict, ui_state: dict):
    """Called when BodiesPanel notifies a change."""
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
    # Auto-calc fluid domain from STL bounding boxes
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
    """Read STL bounding boxes, return (Lx, Ly, Lz) in meters with 20% buffer. Runs off-thread."""
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
                bnd = mesh.bounds  # (xmin,xmax,ymin,ymax,zmin,zmax) in mm
                xmin = min(xmin, bnd[0]); xmax = max(xmax, bnd[1])
                ymin = min(ymin, bnd[2]); ymax = max(ymax, bnd[3])
                zmin = min(zmin, bnd[4]); zmax = max(zmax, bnd[5])
                found = True
            except Exception:
                pass
        if not found:
            return None, None, None
        # Add 20% buffer on each side (total 40% larger in each dimension)
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
    # Launch native PyVista window — same approach as full viewer, no PNG
    from backend.runner import launch_preview_thread
    launch_preview_thread(params)
    ui_state["prev_log_len"] = 0
    ui_state["_prev_timer"].activate()
    ui.notify("Domain preview building in background — window will open automatically.", type="info", timeout=3000)


async def _on_run_click(storage: dict, ui_state: dict):
    if ui_state["sim_running"]:
        # Stop
        abort_sim()
        log = ui_state.get("log_area")
        if log: log.push("⏹  Abort requested…")
        return

    bodies = storage.get("bodies", [])
    if not bodies or not bodies[0].get("stl_path", ""):
        ui.notify("Add at least one solid body with an STL file.", type="warning"); return

    # Confirmation dialog
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
    dom  = dict(Lx=storage.get("Lx",5), Ly=storage.get("Ly",1), Lz=storage.get("Lz",1))
    dx   = storage.get("dx_mm", 10.0)
    nx   = round(dom["Lx"]*1000/dx); ny = round(dom["Ly"]*1000/dx); nz = round(dom["Lz"]*1000/dx)
    n_b  = len(storage.get("bodies", []))
    proj = storage.get("project_name","cht_result")

    with ui.dialog() as dlg, ui.card().classes(
        "bg-neutral-800 border border-neutral-600 min-w-80 gap-3"
    ):
        ui.label("Start Simulation?").classes("text-base font-bold text-slate-200")
        ui.separator().classes("bg-neutral-600")
        with ui.grid(columns=2).classes("gap-x-6 gap-y-1"):
            for label, value in [
                ("Grid",       f"{nx} × {ny} × {nz}  ({nx*ny*nz/1e6:.2f} M cells)"),
                ("Bodies",     str(n_b)),
                ("Voxel",      f"{dx:.1f} mm"),
                ("Flow",       f"{storage.get('flow_dir','+X')}  @  {storage.get('u_in',1.0):.4g} m/s"),
                ("Max iters",  f"{int(storage.get('max_outer',50000)):,}"),
                ("Project",    proj),
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

    # Reset every key from DEFAULT_STATE
    for k, v in DEFAULT_STATE.items():
        storage[k] = copy.deepcopy(v)

    # Reset project name input in header
    proj_inp = ui_state.get("proj_input")
    if proj_inp:
        proj_inp.value = storage["project_name"]

    # Reload page so all tab widgets re-read from fresh storage
    ui.navigate.reload()
    ui.notify("New model created.", type="positive")


async def _save_model_dialog(storage: dict, ui_state: dict):
    from backend.runner import _PROJECTS_DIR

    default_name = storage.get("project_name", "untitled") + ".chtmdl"

    if _is_headless():
        # Headless: save to projects/ dir and show path in notification
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
    """
    Build a server-side file picker widget inside the current UI context.
    Returns a dict ref that will contain {"path": str} when a file is selected.
    `extensions` e.g. [".chtmdl", ".json"]
    """
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
        server_sel: dict = {}  # populated only in headless mode

        if headless:
            # ── Server-side file picker ────────────────────────────────────────
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

    # Prefer server-side selection if headless and something was picked
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
            # Flatten sim_results/<project>/*.npz into one list sorted by mtime
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
                        size_mb  = f.stat().st_size / 1_048_576
                        mtime    = __import__("datetime").datetime.fromtimestamp(
                            f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                        rel      = f.relative_to(_SIM_OUT_DIR)
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

    # Server-side selection takes priority when headless
    if server_sel.get("path"):
        await _load_npz_into_viewer(server_sel["path"], storage, ui_state,
                                     name=server_sel.get("name", ""))
    elif loaded.get("path"):
        await _load_npz_into_viewer(loaded["path"], storage, ui_state,
                                     name=loaded.get("name", ""))


async def _load_npz_into_viewer(
    npz_path: str, storage: dict, ui_state: dict, name: str = ""
):
    import numpy as np

    # Prevent concurrent loads — if already loading, drop this call
    if ui_state.get("_results_loading"):
        return
    ui_state["_results_loading"] = True

    try:
        await _load_npz_into_viewer_impl(npz_path, storage, ui_state, name)
    finally:
        ui_state["_results_loading"] = False


async def _load_npz_into_viewer_impl(
    npz_path: str, storage: dict, ui_state: dict, name: str = ""
):
    import numpy as np

    storage["result_npz"]      = npz_path
    storage["vis_npz"]         = npz_path

    # Read metadata from NPZ
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
        names = []; flow_d = storage.get("flow_dir","+X"); fmin = 0.0; fmax = 1.0

    populate_vis_bodies(storage, names, ui_state.get("vis_refs", {}))

    sl  = ui_state.get("status_lbl")
    lbl = ui_state.get("res_lbl")
    vis_npz_lbl = ui_state.get("vis_refs", {}).get("npz_lbl")
    if vis_npz_lbl: vis_npz_lbl.set_text(name or Path(npz_path).name)
    if sl: sl.set_text("Exporting temperature surfaces…")
    if lbl: lbl.set_text(f"Exporting {name or Path(npz_path).name}…")
    ui.notify("Exporting results to GLTF…", type="info", timeout=3000)

    # Export per-body temperature-colored GLTF files (off main thread)
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
    legend = []
    for bm in body_list:
        if bm.get("t_min") is not None and bm.get("name") != "Fluid domain":
            mean_t = (bm["t_min"] + bm["t_max"]) / 2.0
            mc     = cmap(float(max(0, min(1, (mean_t - fmin) / max(fmax - fmin, 1e-6)))))
            hx     = "#{:02x}{:02x}{:02x}".format(int(mc[0]*255),int(mc[1]*255),int(mc[2]*255))
            legend.append(dict(name=f"{bm['name']} {mean_t:.1f}°C", color=hx))

    # Build colorbar ticks for overlay
    cbar_ticks = []
    for i in range(6):
        t = fmin + i*(fmax-fmin)/5.0
        mc = cmap(i/5.0)
        hx = "#{:02x}{:02x}{:02x}".format(int(mc[0]*255),int(mc[1]*255),int(mc[2]*255))
        cbar_ticks.append(dict(label=f"{t:.1f}", color=hx))
    ui_state["result_cbar"]      = dict(ticks=cbar_ticks, field=field,
                                          fmin=fmin, fmax=fmax,
                                          cmap=export_out.get("cmap","inferno"))
    ui_state["result_grid_size"] = export_out.get("grid_size", [100, 100, 100])

    # Merge streamlines into body list for unified GLTF loading
    all_gltf = list(body_list)
    if export_out.get("streamlines"):
        all_gltf.append(dict(name="Streamlines",
                              gltf_url=export_out["streamlines"],
                              is_lines=True))

    # Cache-bust: append timestamp so browser re-fetches overwritten GLTF files
    import time as _time
    _ts = int(_time.time())
    for _item in all_gltf:
        if "gltf_url" in _item:
            base = _item["gltf_url"].split("?")[0]
            _item["gltf_url"] = f"{base}?v={_ts}"

    ui_state["result_meta"]   = all_gltf
    ui_state["result_legend"] = legend

    _show_viewer_mode(ui_state, "results")  # calls scene_refresh internally

    if lbl: lbl.set_text(f"{name or Path(npz_path).name}  {field}: {fmin:.1f}–{fmax:.1f}")
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

    vis_bodies = storage.get("vis_bodies", {})
    show_bodies = [n for n, v in vis_bodies.items() if v] or None

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

            # Table
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

    # Schedule on main event loop via ui.timer (has correct NiceGUI slot context)
    ui.timer(0.1, _show, once=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ in {"__main__", "__mp_main__"}:
    # Raise upload limit to 100 MB for large STL files.
    # Starlette's MultiPartParser caps each part at 1 MB by default,
    # causing large uploads to stall. Patch at class level before ui.run().
    try:
        from starlette.formparsers import MultiPartParser
        MultiPartParser.max_part_size = 100 * 1024 * 1024  # 100 MB
    except Exception:
        pass

    ui.run(
        title="FluxCore3D",
        dark=True,
        port=8080,
        reload=False,
        storage_secret="fluxcore3d-secret-2024",
        favicon="⚡",
    )
