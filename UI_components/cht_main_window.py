"""
cht_main_window.py  (v4 — updated for Domains/BCs tab restructure)
"""
import traceback
from pathlib import Path
import numpy as np

from PyQt5.QtWidgets import QMainWindow, QSplitter, QApplication, QFileDialog
from PyQt5.QtCore    import Qt, QThread

from UI_components.cht_constants   import SOLID_PRESETS
from UI_components.cht_models      import build_state, save_model, load_model
from UI_components.cht_widgets     import ControlPanel, ViewerPanel
from UI_components.cht_worker      import SimWorker
from UI_components.cht_sim_imports import (
    HAS_SIM, SIM_IMPORT_ERROR,
    SimConfig3D, SolidPropsSI, STLVoxelizer, LBMCHT3D_Torch,
    patch_stl_voxelizer, unpatch_stl_voxelizer, SolidAssembly, SurfaceFluxBC, VolumetricHeatBC,
)

def launch_results_viewer(params: dict, flow_dir: str = "+X", plotter=None, dark: bool = True, low_quality: bool = False):
    import pyvista as pv
    import numpy as np

    npz_path      = params["npz_path"]
    field         = params["field"]
    cmap          = params["cmap"]
    show_bodies   = params["show_bodies"]
    show_fluid    = params["show_fluid"]
    fluid_opacity = params["fluid_opacity"]
    do_stream     = params["streamlines"]
    n_seeds       = params["n_seeds"]
    seed_mode     = params["seed_mode"]

    if not npz_path or not Path(npz_path).exists():
        print("[Visualize] No valid NPZ path."); return

    data = np.load(npz_path, allow_pickle=True)

    if field not in data.files:
        print(f"[Visualize] Field '{field}' not in NPZ. Available: {data.files}"); return
    arr = data[field]
    F = (arr[0] if arr.ndim == 4 else arr).astype(np.float32)
    nx, ny, nz = F.shape

    fluid_k0   = int(data["fluid_k0"]) if "fluid_k0" in data.files else 0
    body_names = list(data["body_names"]) if "body_names" in data.files else []
    body_masks = {}
    for name in body_names:
        key = f"solid_{name}"
        if key in data.files:
            m = data[key].astype(bool)
            if m.ndim == 4: m = m[0]
            body_masks[name] = m

    if show_bodies is None:
        show_bodies = list(body_masks.keys())

    solid_union = np.zeros((nx,ny,nz), dtype=bool)
    for m in body_masks.values(): solid_union |= m
    solid_union_for_fluid = solid_union.copy()
    if fluid_k0 > 0: solid_union_for_fluid[:,:,:fluid_k0] = True

    axis_map  = {"+X":0,"-X":0,"+Y":1,"-Y":1,"+Z":2,"-Z":2}
    flow_axis = axis_map.get(flow_dir, 0)

    sample_vals = []
    for name in show_bodies:
        if name in body_masks:
            sample_vals.append(F[body_masks[name]])
    if show_fluid:
        fluid_mask = ~solid_union_for_fluid
        sample_vals.append(F[fluid_mask & np.isfinite(F)])
    filtered = [v[np.isfinite(v)] for v in sample_vals if len(v) > 0]
    filtered = [v for v in filtered if len(v) > 0]
    if filtered:
        all_v = np.concatenate(filtered)
        fmin, fmax = float(all_v.min()), float(all_v.max())
    else:
        fmin, fmax = float(np.nanmin(F)), float(np.nanmax(F))
    if fmin == fmax: fmax = fmin + 1.0

    if plotter is not None:
        pl = plotter; pl.clear()
    else:
        pl = pv.Plotter(window_size=(1400,900), title=f"CHT Results — {field}")

    bg_col  = "#0D1117" if dark else "#F4F6FA"
    txt_col = "white"   if dark else "black"
    pl.set_background(bg_col)

    if low_quality:
        try: pl.renderer.SetUseFXAA(False)
        except Exception: pass
        try: pl.disable_anti_aliasing()
        except Exception: pass

    sbar_args = dict(title=field, n_labels=5, fmt="%.4g", vertical=True,
                     position_x=0.91, position_y=0.10, width=0.04, height=0.75,
                     title_font_size=18, label_font_size=14, color=txt_col)
    added_sbar = [False]

    if show_bodies:
        step = 2 if low_quality else 1
        Fx = F[::step, ::step, ::step]
        snx, sny, snz = Fx.shape
        grid_pts = pv.ImageData(dimensions=(snx,sny,snz), spacing=(step,step,step), origin=(0,0,0))
        grid_pts.point_data["scalars"] = Fx.reshape(-1, order="F").astype(np.float32)
        for name in show_bodies:
            if name not in body_masks: continue
            mask = body_masks[name][::step, ::step, ::step]
            body_grid = pv.ImageData(dimensions=(snx,sny,snz), spacing=(step,step,step), origin=(0,0,0))
            body_grid.point_data["mask"] = mask.reshape(-1, order="F").astype(np.uint8)
            surf = body_grid.contour([0.5], scalars="mask").sample(grid_pts)
            if surf.n_points == 0: continue
            pl.add_mesh(surf, scalars="scalars", cmap=cmap, clim=(fmin,fmax),
                        opacity=1.0, show_scalar_bar=not added_sbar[0],
                        scalar_bar_args=sbar_args if not added_sbar[0] else {},
                        smooth_shading=False)
            added_sbar[0] = True

    if show_fluid:
        fluid_mask = ~solid_union_for_fluid
        step = 2 if low_quality else 1
        F_sub  = F[::step, ::step, ::step].astype(np.float32)
        fm_sub = fluid_mask[::step, ::step, ::step]
        fnx, fny, fnz = F_sub.shape
        F_vol = F_sub.copy(); F_vol[~fm_sub] = np.nan
        grid_vol = pv.ImageData(dimensions=(fnx,fny,fnz), spacing=(step,step,step), origin=(0,0,0))
        grid_vol.point_data["scalars"] = F_vol.reshape(-1, order="F")
        op_tf = np.full(256, fluid_opacity, dtype=np.float32); op_tf[0] = 0.0
        vol = pl.add_volume(grid_vol, scalars="scalars", clim=(fmin, fmax),
                            cmap=cmap, opacity=op_tf, show_scalar_bar=False, mapper="smart")
        try: vol.mapper.SetAutoAdjustSampleDistances(False)
        except Exception: pass
        try: vol.mapper.SetSampleDistance(1.0 if low_quality else 0.5)
        except Exception: pass
        try: vol.GetProperty().SetScalarOpacityUnitDistance(float(step))
        except Exception:
            try: vol.prop.SetScalarOpacityUnitDistance(float(step))
            except Exception: pass
        if not added_sbar[0]:
            _dummy = pv.Sphere(radius=0.0001, center=(fnx*step/2, fny*step/2, fnz*step/2))
            _dummy["_v"] = np.linspace(fmin, fmax, _dummy.n_points, dtype=np.float32)
            pl.add_mesh(_dummy, scalars="_v", clim=(fmin,fmax), cmap=cmap,
                        opacity=0.0, show_scalar_bar=True, scalar_bar_args=sbar_args)
            added_sbar[0] = True

    if do_stream and all(k in data.files for k in ("u","v","w")):
        u = (data["u"][0] if data["u"].ndim==4 else data["u"]).astype(np.float32)
        v = (data["v"][0] if data["v"].ndim==4 else data["v"]).astype(np.float32)
        w = (data["w"][0] if data["w"].ndim==4 else data["w"]).astype(np.float32)
        gv = pv.ImageData(dimensions=(nx,ny,nz), spacing=(1,1,1), origin=(0,0,0))
        gv.point_data["U"] = np.stack([u.reshape(-1,order="F"), v.reshape(-1,order="F"), w.reshape(-1,order="F")], axis=1)
        gv.set_active_vectors("U")
        rng = np.random.default_rng(42)
        if seed_mode == "inlet_plane":
            idx = 2
            if flow_axis == 0:
                coords = np.argwhere(~solid_union[idx,:,:]); chosen = coords[rng.integers(0,len(coords),n_seeds)]
                pts = np.stack([np.full(n_seeds,idx), chosen[:,0], chosen[:,1]],1).astype(np.float32)
            elif flow_axis == 1:
                coords = np.argwhere(~solid_union[:,idx,:]); chosen = coords[rng.integers(0,len(coords),n_seeds)]
                pts = np.stack([chosen[:,0], np.full(n_seeds,idx), chosen[:,1]],1).astype(np.float32)
            else:
                coords = np.argwhere(~solid_union[:,:,idx]); chosen = coords[rng.integers(0,len(coords),n_seeds)]
                pts = np.stack([chosen[:,0], chosen[:,1], np.full(n_seeds,idx)],1).astype(np.float32)
        else:
            pts = rng.random((n_seeds,3)).astype(np.float32) * np.array([nx,ny,nz])
        seeds = pv.PolyData(pts)
        sl = gv.streamlines_from_source(seeds, vectors="U", integration_direction="forward", initial_step_length=1.5, terminal_speed=0.0,
                max_steps=5000 if low_quality else 100000, integrator_type=45)
        if sl.n_points > 0 and "U" in sl.point_data:
            spd = np.linalg.norm(sl.point_data["U"], axis=1).astype(np.float32)
            sl["speed"] = spd
            pl.add_mesh(sl, scalars="speed", cmap="Spectral", line_width=2,
                        clim=(float(spd.min()), float(spd.max())+1e-10), show_scalar_bar=False)

    pl.add_axes(color=txt_col)
    pl.add_text(f"{npz_path.replace(chr(92),'/').split('/')[-1]}  |  field={field}  |  flow={flow_dir}",
                position="upper_left", font_size=9, color=txt_col)
    if plotter is None: pl.show()
    else: pl.reset_camera(); pl.render()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CHT Workbench  \u2014  Multi-Body Conjugate Heat Transfer")
        self.resize(1380, 840); self.setMinimumSize(900, 600)
        self._running = False; self._sim_thread = None; self._sim_worker = None
        self._result_npz = None; self._result_flow_dir = "+X"

        splitter = QSplitter(Qt.Horizontal); splitter.setHandleWidth(3)
        splitter.setStyleSheet("QSplitter::handle { background:#1A2236; }")
        self.setCentralWidget(splitter)
        self.ctrl   = ControlPanel()
        self.viewer = ViewerPanel()
        splitter.addWidget(self.ctrl); splitter.addWidget(self.viewer)
        splitter.setStretchFactor(0,0); splitter.setStretchFactor(1,1); splitter.setSizes([520,860])
        self.statusBar().showMessage("Ready")

        self.ctrl.btn_preview.clicked.connect(self._preview_geometry)
        self.ctrl.btn_run.clicked.connect(self._toggle_simulation)
        self.ctrl.domains_tab.save_model_clicked.connect(self._save_model)
        self.ctrl.domains_tab.load_model_clicked.connect(self._load_model)
        self.viewer.load_results_clicked.connect(self._load_results)
        self.ctrl.visualize_tab.btn_plot.clicked.connect(self._launch_viz)

        if not HAS_SIM and SIM_IMPORT_ERROR is not None:
            self.ctrl.log_msg("[WARN] Simulation modules failed to load:")
            for line in str(SIM_IMPORT_ERROR).splitlines():
                self.ctrl.log_msg(f"       {line}")

    # ── convenience accessors ─────────────────────────────────────────────────
    @property
    def _dt(self): return self.ctrl.domains_tab
    @property
    def _bt(self): return self.ctrl.bcs_tab

    # ── Save / Load model ─────────────────────────────────────────────────────

    def _save_model(self):
        p = self.ctrl.get_all_params()
        project_name = self.viewer.project_name
        path, _ = QFileDialog.getSaveFileName(self, "Save Model",
            f"{project_name or 'untitled'}.chtmdl", "CHT Model Files (*.chtmdl);;All Files (*)")
        if not path: return
        state = build_state(
            domain=p["domain"], j0_divisor=p["j0_divisor"], bodies=p["bodies"],
            fluid=p["fluid"], t_in_C=p["t_in_C"], t_amb_C=p["t_amb_C"],
            flow_dir=p["flow_dir"], u_in=p["u_in"],
            collision=p["collision"], outlet_bc=p["outlet_bc"],
            bc_type=p["bc_type"], bc_params=p["bc_params"],
            dx_mm=p["dx_mm"], tol_u_ema=p["tol_u_ema"],
            max_outer=int(p["max_outer"]), tol_dTs=p["tol_dTs"],
            tol_dTf=p["tol_dTf"], dt_scale_max=p["dt_scale_max"],
            max_mg=int(p["max_mg"]), project_name=project_name,
            gpu_raytrace=p.get("gpu_raytrace", True),
            face_states    = p.get("face_states", {}),
            domain_walls   = p.get("domain_walls",  []),
            domain_outlets = p.get("domain_outlets", []))
        
        if save_model(state, path):
            self.ctrl.log_msg(f"\U0001f4be  Model saved: {path}")
            self.statusBar().showMessage(f"Model saved: {Path(path).name}")
        else:
            self.ctrl.log_msg(f"[ERROR] Could not save model to {path}")

    def _load_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Model", "",
            "CHT Model Files (*.chtmdl);;All Files (*)")
        if not path: return
        state = load_model(path)
        if state is None:
            self.ctrl.log_msg(f"[ERROR] Could not load model from {path}"); return
        self.ctrl.restore(state)
        proj = state.get("project_name", "").strip()
        if proj: self.viewer.txt_project.setText(proj)
        self.ctrl.log_msg(f"\U0001f4c2  Model loaded: {path}")
        self.statusBar().showMessage(f"Model loaded: {Path(path).name}")

    # ── Load existing results ─────────────────────────────────────────────────

    def _load_results(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Results", "",
            "NumPy Archive (*.npz);;All Files (*)")
        if not path: return
        self.ctrl.log_msg(f"\U0001f4c2  Loading results: {path}")
        self.statusBar().showMessage(f"Loading: {Path(path).name}")
        QApplication.processEvents()
        try:
            data     = np.load(path, allow_pickle=True)
            flow_dir = str(data["flow_dir"]) if "flow_dir" in data.files else self._bt.cmb_dir.currentText()
        except Exception:
            flow_dir = self._bt.cmb_dir.currentText()
        self._result_npz = path; self._result_flow_dir = flow_dir
        self.ctrl.visualize_tab.set_npz(path)
        self.viewer.view_tabs.setCurrentIndex(1)
        self.viewer.set_info(f"Results: {Path(path).name}")
        self.viewer.enable_full_viewer(True)
        try: self.viewer._btn_fullviewer.clicked.disconnect()
        except Exception: pass
        self.viewer._btn_fullviewer.clicked.connect(lambda: self._launch_full_viewer(path, flow_dir))
        self.viewer.vp_res.show_results_embedded(path, flow_dir, cmap="inferno")
        self.ctrl.log_msg(f"\u2705  Results loaded: {Path(path).name}  |  flow_dir={flow_dir}")
        self.statusBar().showMessage(f"Results loaded: {Path(path).name}")

    # ── Geometry preview ──────────────────────────────────────────────────────

    def _preview_geometry(self):
        bodies = self._dt.bodies_panel.bodies
        if not bodies:
            self.statusBar().showMessage("Add at least one solid body first."); return
        if not HAS_SIM:
            self.ctrl.log_msg("[WARN] Simulation modules absent \u2014 cannot preview."); return

        dom      = self._dt.domain
        dx_mm    = self._bt.spn_dx.value()
        nx       = round(dom["Lx"]*1000/dx_mm)
        ny       = round(dom["Ly"]*1000/dx_mm)
        nz       = round(dom["Lz"]*1000/dx_mm)
        flow_dir = self._bt.cmb_dir.currentText()
        j0_div   = self._dt.j0_divisor
        bct      = self._bt.bc_type
        bcp      = self._bt.bc_params

        self.ctrl.log_msg(f"\u2bc1  Preview  flow={flow_dir}  j0={j0_div}  grid={nx}\u00d7{ny}\u00d7{nz} \u2026")
        self.viewer.set_info("Building assembly \u2026"); QApplication.processEvents()

        try:
            if self._dt.chk_gpu_rt.isChecked():
                patch_stl_voxelizer(STLVoxelizer)
            else:
                unpatch_stl_voxelizer(STLVoxelizer)
            cfg_prev = SimConfig3D(
                Lx_m=dom["Lx"], Ly_m=dom["Ly"], Lz_m=dom["Lz"],
                flow_bc="inlet_outlet", flow_dir=flow_dir, transverse_walls=True,
                u_in_mps=self._bt.spn_uin.value(),
                dt_thermal_phys_s=1.0, temp_bc="fixed", t_ambient_C=20.0,
                heating_mode="off", qdot_total_W=0.0,
                solid_init_mode="ambient", T_hot_C=0.0,
            )
            cfg_prev.dx_mm=dx_mm; cfg_prev.nx=nx; cfg_prev.ny=ny; cfg_prev.nz=nz; cfg_prev.obstacle=None

            bodies_dicts = [
                dict(stl=b.stl_path, npz=b.npz_name, name=b.name, build_dir=b.build_dir,
                     material=SolidPropsSI(**SOLID_PRESETS["Aluminum (LBM-scaled)"]),
                     color=b.color, role=b.role)
                for b in bodies
            ]
            assembly = SolidAssembly(
                bodies=bodies_dicts, fluid_Lz_m=dom["Lz"], dx_mm=dx_mm, cfg=cfg_prev,
                flow_dir=flow_dir, j0_divisor=j0_div,
                STLVoxelizer_cls=STLVoxelizer, LBMCHT_cls=LBMCHT3D_Torch,
            )
            assembly.build_domain()

            flow_ax = {"X":0,"Y":1,"Z":2}[flow_dir[1]]
            body_data = []
            for b, spec in zip(bodies, assembly.specs):
                off   = spec.offset_ijk
                n_vox = int(np.load(b.npz_name, allow_pickle=True)["voxel_solid"].sum())
                self.ctrl.log_msg(f"   {b.name}: {n_vox:,} cells  offset=({off[0]},{off[1]},{off[2]})  [flow-axis:{off[flow_ax]}]")
                body_data.append(dict(name=b.name, npz=b.npz_name, offset_ijk=off, color=b.color))

            flux_bcs_preview = []; vol_bcs_preview = []
            if bct != "off" and bcp.get("solid_name"):
                tgt  = bcp["solid_name"]
                spec = next((s for s in assembly.specs if s.name == tgt), None)
                if spec is not None:
                    try:
                        vox      = np.load(spec.name+".npz", allow_pickle=True)["voxel_solid"].astype(bool)
                        dom_mask = np.zeros((nx,ny,nz), dtype=bool)
                        i0,j0,k0 = (int(x) for x in spec.offset_ijk); Gx,Gy,Gz=vox.shape
                        di0=max(i0,0);di1=min(i0+Gx,nx); dj0=max(j0,0);dj1=min(j0+Gy,ny); dk0=max(k0,0);dk1=min(k0+Gz,nz)
                        si0=di0-i0;si1=si0+(di1-di0); sj0=dj0-j0;sj1=sj0+(dj1-dj0); sk0=dk0-k0;sk1=sk0+(dk1-dk0)
                        dom_mask[di0:di1,dj0:dj1,dk0:dk1]=vox[si0:si1,sj0:sj1,sk0:sk1]
                        n_vox_bc=int(dom_mask.sum())
                        if bct == "surface_flux":
                            import torch as _torch
                            bc_prev = SurfaceFluxBC(
                                solid_mask   = _torch.from_numpy(dom_mask),
                                axis         = bcp.get("axis", "-Z"),
                                surface_L_mm = bcp.get("L_mm", 400.0),
                                surface_W_mm = bcp.get("W_mm", 200.0),
                                q_flux_W_m2  = bcp.get("q_flux", 1800.0),
                                dx_mm        = dx_mm,
                                center_mm    = None,
                            )
                            surface_mask_np = bc_prev.voxel_mask.numpy()
                            n_vox_bc = int(surface_mask_np.sum())
                            flux_bcs_preview.append(dict(mask_np=surface_mask_np,
                                axis=bcp.get("axis","-Z"), q=bcp.get("q_flux",0.0), n_vox=n_vox_bc))
                            self.ctrl.log_msg(f"   \U0001f525 SurfaceFlux preview: {n_vox_bc:,} surface cells  (axis={bcp.get('axis','-Z')})")
                        else:
                            vol_bcs_preview.append(dict(mask_np=dom_mask, name=tgt, Q=bcp.get("Q_watts",0), n_vox=n_vox_bc))
                            self.ctrl.log_msg(f"   \U0001f525 VolumetricHeat: {n_vox_bc:,} cells")
                    except Exception as exc:
                        self.ctrl.log_msg(f"   [WARN] BC preview failed: {exc}")

            self.viewer.vp_geo.show_domain(
                nx=nx, ny=ny, nz=nz, dx_mm=dx_mm,
                body_data=body_data, flux_bcs=flux_bcs_preview, vol_bcs=vol_bcs_preview,
                flow_dir=flow_dir,
                annotation=(f"Domain {nx*dx_mm:.0f}\u00d7{ny*dx_mm:.0f}\u00d7{nz*dx_mm:.0f} mm"
                             f"  |  {dx_mm:.1f} mm voxel  |  flow {flow_dir}  j0={j0_div}"
                             f"  |  {len(bodies)} bod{'y' if len(bodies)==1 else 'ies'}"),
            )
            self.viewer.view_tabs.setCurrentIndex(0)
            self.viewer.set_info(f"Domain: {nx}\u00d7{ny}\u00d7{nz}  |  {len(bodies)} bod{'y' if len(bodies)==1 else 'ies'}")
            self.statusBar().showMessage(f"Preview ready \u2014 {nx}\u00d7{ny}\u00d7{nz}")

        except Exception as exc:
            self.ctrl.log_msg(f"[ERROR] Preview failed: {exc}")
            self.ctrl.log_msg(traceback.format_exc()[-800:])
            self.statusBar().showMessage("Preview failed \u2014 see log.")

    # ── Simulation ────────────────────────────────────────────────────────────

    def _toggle_simulation(self):
        if self._running: self._abort_sim()
        else: self._start_sim()

    def _start_sim(self):
        if not HAS_SIM:
            self.ctrl.log_msg("[ERROR] Simulation modules not installed."); return
        bodies = self._dt.bodies_panel.bodies
        if not bodies or not bodies[0].stl_path:
            self.statusBar().showMessage("Add at least one solid body with an STL file."); return

        # ── Confirmation dialog ───────────────────────────────────────────
        from PyQt5.QtWidgets import QMessageBox
        p = self.ctrl.get_all_params()
        dom = p["domain"]; dx = p["dx_mm"]
        nx = round(dom["Lx"]*1000/dx); ny = round(dom["Ly"]*1000/dx); nz = round(dom["Lz"]*1000/dx)
        n_bodies = len(bodies)
        msg = QMessageBox(self)
        msg.setWindowTitle("Confirm Simulation")
        msg.setIcon(QMessageBox.Question)
        msg.setText("<b>Start simulation?</b>")
        msg.setInformativeText(
            f"Grid:      {nx} × {ny} × {nz}  ({nx*ny*nz/1e6:.2f} M cells)\n"
            f"Bodies:    {n_bodies}\n"
            f"Voxel:     {dx:.1f} mm\n"
            f"Flow:      {p['flow_dir']}  @  {p['u_in']:.4g} m/s\n"
            f"Max iters: {int(p['max_outer']):,}\n"
            f"Project:   {self.viewer.project_name or 'cht_result'}"
        )
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        msg.button(QMessageBox.Ok).setText("▶   Run")
        msg.button(QMessageBox.Cancel).setText("Cancel")
        msg.setDefaultButton(QMessageBox.Ok)
        if msg.exec_() != QMessageBox.Ok:
            return
        # ─────────────────────────────────────────────────────────────────

        self._running = True
        btn = self.ctrl.btn_run
        btn.setText("\u23f9   Stop"); btn.setProperty("running","true")
        btn.style().unpolish(btn); btn.style().polish(btn)
        self.ctrl.btn_preview.setEnabled(False)
        self.ctrl.progress_bar.setValue(0)

        params = self.ctrl.get_all_params()
        params["project_name"] = self.viewer.project_name or "cht_result"
        self._sim_worker = SimWorker(params)
        self._sim_thread = QThread()
        self._sim_worker.moveToThread(self._sim_thread)
        self._sim_thread.started.connect(self._sim_worker.run)
        self._sim_worker.progress.connect(self._on_progress)
        self._sim_worker.pct.connect(self.ctrl.progress_bar.setValue)
        self._sim_worker.finished.connect(self._on_finished)
        self._sim_worker.error.connect(self._on_error)
        self._sim_worker.finished.connect(self._sim_thread.quit)
        self._sim_worker.error.connect(self._sim_thread.quit)
        self._sim_thread.finished.connect(self._sim_cleanup)
        self.ctrl.log_msg(f"\u25b6  Simulation started  project={params['project_name']} \u2026")
        self._sim_thread.start()

    def _abort_sim(self):
        if self._sim_worker: self._sim_worker.abort()
        self.statusBar().showMessage("Aborting \u2026")

    def _sim_cleanup(self):
        self._running = False
        btn = self.ctrl.btn_run
        btn.setText("\u25b6   Run Model"); btn.setProperty("running","false")
        btn.style().unpolish(btn); btn.style().polish(btn)
        self.ctrl.btn_preview.setEnabled(bool(self._dt.bodies_panel.bodies))

    def _on_progress(self, msg):
        self.ctrl.log_msg(msg); self.ctrl.lbl_status.setText(msg[:60])
        self.statusBar().showMessage(msg)

    def _on_finished(self, npz_path):
        self._result_npz      = npz_path
        self._result_flow_dir = self._bt.cmb_dir.currentText()
        self.ctrl.visualize_tab.set_npz(npz_path)
        self.ctrl.log_msg(f"\u2705  Results saved: {npz_path}")
        self.statusBar().showMessage(f"Complete \u2014 {npz_path}")
        # ── Thermal diagnostics popup ─────────────────────────────────────
        try:
            from UI_components.cht_widgets import ThermalDiagnosticsDialog
            rows = self._build_diagnostics_rows(npz_path)
            if rows:
                dlg = ThermalDiagnosticsDialog(rows, parent=self)
                dlg.show()   # non-blocking — user can dismiss while inspecting results
        except Exception as exc:
            self.ctrl.log_msg(f"[WARN] Diagnostics popup failed: {exc}")
        # ─────────────────────────────────────────────────────────────────        
        self.viewer.view_tabs.setCurrentIndex(1)
        self.viewer.set_info(f"Results: {Path(npz_path).name}")
        self.viewer.enable_full_viewer(True)
        try: self.viewer._btn_fullviewer.clicked.disconnect()
        except Exception: pass
        self.viewer._btn_fullviewer.clicked.connect(lambda: self._launch_full_viewer(npz_path, self._result_flow_dir))
        self.viewer.vp_res.show_results_embedded(npz_path, self._result_flow_dir, cmap="inferno")

    def _build_diagnostics_rows(self, npz_path):
            """Reconstruct thermal diagnostics from saved NPZ — mirrors print_thermal_diagnostics."""
            import numpy as np
            data = np.load(npz_path, allow_pickle=True)
            if "T" not in data.files:
                return None
            arr = data["T"]
            T = (arr[0] if arr.ndim == 4 else arr).astype(np.float32)

            body_names = list(data["body_names"]) if "body_names" in data.files else []
            rows = []
            solid_union = np.zeros(T.shape, dtype=bool)

            for name in body_names:
                key = f"solid_{name}"
                if key not in data.files:
                    rows.append(dict(name=name, tmin="—", tmean="—", tmax="—", nvox="—"))
                    continue
                m = data[key].astype(bool)
                if m.ndim == 4: m = m[0]
                solid_union |= m
                if not m.any():
                    rows.append(dict(name=name, tmin="—", tmean="—", tmax="—", nvox="—"))
                    continue
                T_b = T[m]
                rows.append(dict(name=name,
                                tmin=float(T_b.min()), tmean=float(T_b.mean()),
                                tmax=float(T_b.max()), nvox=int(m.sum())))

            # Summary rows
            fluid_union = ~solid_union
            T_s = T[solid_union]; T_f = T[fluid_union]
            rows.append(dict(name="ALL SOLID",
                            tmin=float(T_s.min()), tmean=float(T_s.mean()),
                            tmax=float(T_s.max()), nvox=int(solid_union.sum())))
            rows.append(dict(name="ALL FLUID",
                            tmin=float(T_f.min()), tmean=float(T_f.mean()),
                            tmax=float(T_f.max()), nvox=int(fluid_union.sum())))
            return rows

    def _launch_full_viewer(self, npz_path, flow_dir):
        self.ctrl.log_msg("\u25a1  Launching full interactive viewer \u2026")
        try:
            LBMCHT3D_Torch.view_snapshots_pyvista_3d_single_npz(
                npz_path=npz_path, fields=["T","speed","u","v","w","rho"],
                initial_field="T", seed_mode="inlet_plane", use_global_clim=True,
                flow_dir=flow_dir, cmap="inferno")
        except Exception as exc:
            self.ctrl.log_msg(f"[ERROR] Full viewer: {exc}")

    def _launch_viz(self):
        p = self.ctrl.visualize_tab.get_plot_params()
        if not p["npz_path"]:
            self.ctrl.log_msg("[Visualize] Load a results file first."); return
        if not p["show_bodies"] and not p["show_fluid"]:
            self.ctrl.log_msg("[Visualize] Select at least one domain to show."); return
        flow_dir = self._bt.cmb_dir.currentText()
        name = p["npz_path"].replace("\\","/").split("/")[-1]
        self.ctrl.log_msg(f"\U0001f4ca  Plotting: {name}  field={p['field']}")
        QApplication.processEvents()
        dark = self.viewer.vp_res.is_dark()
        launch_results_viewer(p, flow_dir=flow_dir,
                              plotter=self.viewer.vp_res.plotter, dark=dark,
                              low_quality=p.get("low_quality", True))
        self.viewer.view_tabs.setCurrentIndex(1)
        self.viewer.set_info(f"Results: {name}  |  field={p['field']}")

    def _on_error(self, msg):
        self.ctrl.log_msg(f"[ERROR] {msg}")
        self.statusBar().showMessage("Simulation failed \u2014 see log.")

    def closeEvent(self, event):
        if self._running: self._abort_sim()
        try: self.viewer.vp_geo.plotter.close()
        except Exception: pass
        try: self.viewer.vp_res.plotter.close()
        except Exception: pass
        event.accept()
