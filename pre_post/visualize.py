import os
import numpy as np
import trimesh
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from matplotlib.colors import Normalize
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path
from scipy.interpolate import griddata
import pyvista as pv
from datetime import datetime
from trimesh.transformations import rotation_matrix
from scipy.interpolate import LinearNDInterpolator



class AM_Visualizer:
    def __init__(self, filepath, config):
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"[ERROR] Simulation result data file not found: {filepath}")
            
        data = np.load(filepath, allow_pickle=True)
        self.x0 = data["x0"]
        self.x = data["x"]
        self.positions_over_time = data["displacement"].tolist()
        self.von_mises_over_time = data.get("von_mises", None)
        self.elastic_strain_eqv_over_time = data.get("elastic_strain_eqv", None)
        self.volumetric_strain_over_time = data.get("volumetric_strain", None)
        self.stresses_over_time = data.get("stresses", None)
        self.contact_stress_over_time = data.get("contact_stress", None)
        self.repulsion_force_over_time = data.get("repulsion_force", None)
        self.config = data["config"].item()
        self.layer_height = self.config["layer_height"]
        self.build_dir = self.config["build_dir"]
        scale=float(self.config["part_scale"])
        #print("snapshots", len(self.positions_over_time))       

        try:
            self.mesh_path = data["mesh_path"].item()
            self._load_geometry(scale)
            # mesh.apply_scale(self.config["part_scale"])
            # self.mesh = mesh
            # if isinstance(mesh, trimesh.Scene):
            #     self.mesh = trimesh.util.concatenate([g for g in self.mesh.geoemtry.values()])
            # self._center_mesh_to_chamber()            
        except Exception as e:
            raise RuntimeError(f"Failed to load STL: {e}")
        
        current_time = datetime.now().strftime("%I:%M:%S %p")
        print(f"[{current_time}]...Loaded simulation data for visualization")

        if config["disp_vectors"]:
            self._visualize_displacement_vectors()
        if config ["render_3D"]:
            #self._interactive_3Dviewer()
            self.MPM_visualizer()      

    def _load_geometry(self, scale):
        mesh = trimesh.load_mesh(self.mesh_path)
        mesh.apply_scale(scale)
        self.mesh = mesh
        
        if isinstance(mesh, trimesh.Scene):
            self.mesh = trimesh.util.concatenate([g for g in self.mesh.geoemtry.values()])        
        
        # Define rotation axis: X, Y, or Z
        axis = self.build_dir

        if axis== "+X":
            axis_vector = [0, 1, 0]
            angle_deg = -90
        elif axis== "-X":
            axis_vector = [0, 1, 0]
            angle_deg = 90
        elif axis== "+Y":
            axis_vector = [1, 0, 0]
            angle_deg = -90
        elif axis== "-Y":
            axis_vector = [1, 0, 0]
            angle_deg = 90 
        elif axis== "+Z":
            axis_vector = [0, 0, 1]
            angle_deg = 0
        elif axis== "-Z":
            axis_vector = [1, 0, 0]
            angle_deg = 180

        angle_rad = np.deg2rad(angle_deg)
        # Define rotation matrix (about centroid)
        rot_matrix = rotation_matrix(angle_rad, axis_vector, mesh.centroid)

        # Apply rotation
        self.mesh.apply_transform(rot_matrix)

        center_x = np.mean(self.mesh.bounds[:, 0])
        center_y = np.mean(self.mesh.bounds[:, 1])

        # ----- ALIGN Z WITH POINT CLOUD -----
        # Get the point cloud's z reference (from simulation data)
        point_cloud_z_min = self.x0[:, 2].min()
        
        # Get current STL z_min after rotation
        stl_z_min = self.mesh.bounds[0][2]
        
        # Shift STL so its z_min matches point cloud's z_min
        z_correction = point_cloud_z_min - stl_z_min

        self.mesh.apply_translation([-center_x, -center_y, z_correction])

    def MPM_visualizer1(self, gif_path="results/animation/field.gif"):
        os.makedirs(os.path.dirname(gif_path), exist_ok=True)

        plotter = pv.Plotter()
        if self.mesh is not None:
            plotter.add_mesh(pv.wrap(self.mesh), color='gray', opacity=0.2)

        cloud_actor = None
        gif_frames = []
        recording = {"enabled": False}
        current_field = {"name": "Displacement"}  # default
        x0_np = self.x0

        def get_step_data(step, field_name):
            coords = self.positions_over_time[step]
            if field_name == "Displacement":
                scal = np.linalg.norm(coords - x0_np, axis=1)
                name = "Displacement"
            elif field_name == "von Mises":
                if self.von_mises_over_time is not None:
                    scal = self.von_mises_over_time[step]
                else:
                    # on-the-fly if not saved
                    S = self.stresses_over_time[step]
                    sxx, syy, szz = S[:,0,0], S[:,1,1], S[:,2,2]
                    sxy, syz, szx = S[:,0,1], S[:,1,2], S[:,2,0]
                    scal = np.sqrt(0.5*((sxx-syy)**2 + (syy-szz)**2 + (szz-sxx)**2) + 3.0*(2.0*(sxy**2+syz**2+szx**2))/2.0)
                name = "von Mises (MPa)"
            elif field_name == "ε_eq (F)":
                scal = self.elastic_strain_eqv_over_time[step]
                name = "ε_eq (F)"
            elif field_name == "ε_vol":
                scal = self.volumetric_strain_over_time[step]
                name = "ε_vol = J - 1"
            else:
                raise ValueError(f"Unknown field '{field_name}'")
            return coords, scal, name

        def redraw(step):
            nonlocal cloud_actor
            coords, scal, name = get_step_data(step, current_field["name"])
            cloud = pv.PolyData(coords)
            cloud[name] = scal
            if cloud_actor:
                plotter.remove_actor(cloud_actor)
            cloud_actor = plotter.add_mesh(
                cloud, scalars=name, cmap="viridis",
                render_points_as_spheres=True, point_size=8,
                show_scalar_bar=True, clim=[float(np.min(scal)), float(np.max(scal))]
            )
            plotter.update_scalar_bar_range([float(np.min(scal)), float(np.max(scal))])
            plotter.add_text(f"{name}", font_size=10, position='lower_left')
            plotter.render()
            if recording["enabled"]:
                gif_frames.append(plotter.screenshot(transparent_background=False))

        # slider to scrub time
        n_steps = len(self.positions_over_time)
        def slider_callback(value):
            redraw(int(round(value)))
        plotter.add_slider_widget(slider_callback, rng=[0, n_steps-1], value=0, title="Time Step")

        # three checkboxes (mutually exclusive)
        # positions chosen so they don't overlap
        def make_field_toggle(label, pos, field_label):
            def cb(state):
                if state:
                    # turn this on and turn others off visually by resetting their buttons
                    current_field["name"] = field_label
                    # force other buttons off by re-calling them with False
                    for l, btn in buttons.items():
                        if l != field_label:
                            btn.SetState(False)  # pv returns vtkBool; this just flips the UI
                    # redraw | current slider value
                    redraw(int(round(plotter.slider_widgets[0].GetRepresentation().GetValue())))
            btn = plotter.add_checkbox_button_widget(cb, value=(field_label==current_field["name"]),
                                                    position=pos, size=22, color_on="yellow")
            # add text label near it
            px, py = pos
            plotter.add_text(label, position=(px+30, py), font_size=10)
            return btn

        buttons = {}
        buttons["Displacement"] = make_field_toggle("Displacement", (10, 80), "Displacement")
        buttons["von Mises"]    = make_field_toggle("von Mises",    (10, 50), "von Mises")
        buttons["ε_eq (F)"] = make_field_toggle("ε_eq (F)", (10, -10), "ε_eq (F)")
        buttons["ε_vol"]    = make_field_toggle("ε_vol", (10, -40), "ε_vol")

        # recording toggle
        def toggle_recording(val):
            recording["enabled"] = val
            print("Recording:", "ON" if val else "OFF")
        plotter.add_checkbox_button_widget(toggle_recording, value=False, position=(10, 110), size=22, color_on="yellow")
        plotter.add_text("Record GIF", position=(42, 110), font_size=10)

        # initial draw
        redraw(0)
        plotter.show()

        if gif_frames:
            imageio.mimsave(gif_path, gif_frames, duration=0.5)
            print(f"[{datetime.now().strftime('%I:%M:%S %p')}] Saved GIF: {gif_path}")

    def MPM_visualizer(self, gif_path="results/animation/field.gif"):
        os.makedirs(os.path.dirname(gif_path), exist_ok=True)

        plotter = pv.Plotter()
        if self.mesh is not None:
            plotter.add_mesh(pv.wrap(self.mesh), color='gray', opacity=0.2)

        cloud_actor = None
        text_actor = None  # Track text actor for removal
        gif_frames = []
        recording = {"enabled": False}
        current_field = {"name": "Displacement"}
        x0_np = self.x0

        def get_step_data(step, field_name):
            coords = self.positions_over_time[step]
            if field_name == "Displacement":
                scal = np.linalg.norm(coords - x0_np, axis=1)
                name = "Displacement (mm)"
            elif field_name == "von Mises":
                if self.von_mises_over_time is not None:
                    scal = self.von_mises_over_time[step]
                else:
                    S = self.stresses_over_time[step]
                    sxx, syy, szz = S[:,0,0], S[:,1,1], S[:,2,2]
                    sxy, syz, szx = S[:,0,1], S[:,1,2], S[:,2,0]
                    scal = np.sqrt(0.5*((sxx-syy)**2 + (syy-szz)**2 + (szz-sxx)**2) + 3.0*(sxy**2+syz**2+szx**2))
                name = "von Mises (MPa)"
            elif field_name == "ε_eq (F)":
                scal = self.elastic_strain_eqv_over_time[step]
                name = "ε_eq (F)"
            elif field_name == "ε_vol":
                scal = self.volumetric_strain_over_time[step]
                name = "ε_vol = J - 1"
            else:
                raise ValueError(f"Unknown field '{field_name}'")
            return coords, scal, name

        def redraw(step):
            nonlocal cloud_actor, text_actor
            coords, scal, name = get_step_data(step, current_field["name"])
            cloud = pv.PolyData(coords)
            cloud[name] = scal
            
            if cloud_actor is not None:
                plotter.remove_actor(cloud_actor)
            if text_actor is not None:
                plotter.remove_actor(text_actor)
                
            cloud_actor = plotter.add_mesh(
                cloud, scalars=name, cmap="viridis",
                render_points_as_spheres=True, point_size=8,
                show_scalar_bar=True, clim=[float(np.min(scal)), float(np.max(scal))]
            )
            plotter.update_scalar_bar_range([float(np.min(scal)), float(np.max(scal))])
            text_actor = plotter.add_text(f"Field: {name}", font_size=10, position='lower_left')
            plotter.render()
            
            if recording["enabled"]:
                gif_frames.append(plotter.screenshot(transparent_background=False))

        def get_current_step():
            """Get current step from slider."""
            if plotter.slider_widgets:
                return int(round(plotter.slider_widgets[0].GetRepresentation().GetValue()))
            return 0

        # Slider to scrub time
        n_steps = len(self.positions_over_time)
        def slider_callback(value):
            redraw(int(round(value)))
        plotter.add_slider_widget(slider_callback, rng=[0, n_steps-1], value=0, title="Time Step")

        # Use simple field selection - each button just sets its field when clicked
        # No mutual exclusion in UI, but the last clicked one wins
        def make_field_button(label, pos, field_label):
            def cb(state):
                # Always switch to this field when clicked (ignore state)
                current_field["name"] = field_label
                redraw(get_current_step())
            plotter.add_checkbox_button_widget(
                cb, value=(field_label == "Displacement"),  # Only Displacement starts checked
                position=pos, size=22, color_on="yellow"
            )
            px, py = pos
            plotter.add_text(label, position=(px + 30, py), font_size=10)

        make_field_button("Displacement", (10, 80), "Displacement")
        make_field_button("von Mises", (10, 50), "von Mises")
        make_field_button("ε_eq (F)", (10, 20), "ε_eq (F)")
        make_field_button("ε_vol", (10, -10), "ε_vol")

        # Recording toggle
        def toggle_recording(val):
            recording["enabled"] = val
            print("Recording:", "ON" if val else "OFF")
        plotter.add_checkbox_button_widget(toggle_recording, value=False, position=(10, 120), size=22, color_on="red")
        plotter.add_text("Record GIF", position=(42, 120), font_size=10)

        # Initial draw
        redraw(0)
        plotter.show()

        if gif_frames:
            imageio.mimsave(gif_path, gif_frames, duration=0.5)
            print(f"[{datetime.now().strftime('%I:%M:%S %p')}] Saved GIF: {gif_path}")


    def visualize_displacement_evolution_interactive(self, gif_path="results/animation/sph_displacement.gif"):
        plotter = pv.Plotter()
        stl_mesh = pv.wrap(self.mesh)
        cloud_actor = None
        idx = 0
        gif_frames = []
        recording = {"enabled": False}
        os.makedirs(os.path.dirname(gif_path), exist_ok=True)
        x0_np = self.x0

        overlay_mode = {"type": "Displacement"}  # Options: Displacement, Contact_Stress, Repulsion_Force

        def get_step_data(step):
            coords = self.positions_over_time[step]
            displacements = np.linalg.norm(coords - x0_np, axis=1)
            contact_stress = self.contact_stress_over_time[step] if hasattr(self, "contact_stress_over_time") else displacements * 0
            repulsion_force = self.repulsion_force_over_time[step] if hasattr(self, "repulsion_force_over_time") else displacements * 0
            return coords, displacements, contact_stress, repulsion_force

        def update_visualization(step):
            nonlocal cloud_actor
            coords, disp, stress, repulsion = get_step_data(step)

            cloud = pv.PolyData(coords)
            if overlay_mode["type"] == "Displacement":
                cloud["Metric"] = disp
                cmap, label = "viridis", "Displacement"
            elif overlay_mode["type"] == "Contact_Stress":
                cloud["Metric"] = stress
                cmap, label = "inferno", "Contact Stress"
            elif overlay_mode["type"] == "Repulsion_Force":
                cloud["Metric"] = repulsion
                cmap, label = "cividis", "Repulsion Force"

            if cloud_actor:
                plotter.remove_actor(cloud_actor)

            cloud_actor = plotter.add_mesh(cloud, scalars="Metric", cmap=cmap,
                                        render_points_as_spheres=True, point_size=6,
                                        show_scalar_bar=True, clim=[np.min(cloud["Metric"]), np.max(cloud["Metric"])])
            plotter.update_scalar_bar_range([np.min(cloud["Metric"]), np.max(cloud["Metric"])])
            plotter.add_text(label, font_size=10, position='lower_left', name="overlay_label")
            plotter.render()
            if recording["enabled"]:
                gif_frames.append(plotter.screenshot(transparent_background=False))

        def slider_callback(value):
            step = int(round(value))
            update_visualization(step)

        def toggle_recording(value):
            recording["enabled"] = value
            print("Recording:", "ON" if value else "OFF")

        def set_overlay_disp(_): overlay_mode.update(type="Displacement"); update_visualization(idx)
        def set_overlay_stress(_): overlay_mode.update(type="Contact_Stress"); update_visualization(idx)
        def set_overlay_repulsion(_): overlay_mode.update(type="Repulsion_Force"); update_visualization(idx)

        plotter.add_mesh(stl_mesh, color='gray', opacity=0.2)
        plotter.add_slider_widget(slider_callback, rng=[0, len(self.positions_over_time) - 1], value=0, title="Time Step")

        plotter.add_checkbox_button_widget(toggle_recording, value=False, position=(10, 80), size=20, color_on="yellow")
        plotter.add_text("Record GIF", position=(40, 80), font_size=10)

        # Overlay switches
        plotter.add_checkbox_button_widget(set_overlay_disp, value=True, position=(10, 120), size=20)
        plotter.add_text("Displacement", position=(40, 120), font_size=10)

        plotter.add_checkbox_button_widget(set_overlay_stress, value=False, position=(10, 160), size=20)
        plotter.add_text("Contact Stress", position=(40, 160), font_size=10)

        plotter.add_checkbox_button_widget(set_overlay_repulsion, value=False, position=(10, 200), size=20)
        plotter.add_text("Repulsion Force", position=(40, 200), font_size=10)

        update_visualization(idx)
        plotter.show()

        if gif_frames:
            imageio.mimsave(gif_path, gif_frames, duration=0.5)
            print(f"[{datetime.now().strftime('%I:%M:%S %p')}] Saved GIF: {gif_path}")


    def visualize_displacement_evolution_interactive1(self, gif_path="results/animation/sph_displacement.gif"):

        plotter = pv.Plotter()
        stl_mesh = pv.wrap(self.mesh)
        cloud_actor = None
        gif_frames = []
        idx = 0
        recording = {"enabled": False}
        os.makedirs(os.path.dirname(gif_path), exist_ok=True)
        x0_np = self.x0

        def get_step_data(step):
            coords = self.positions_over_time[step]
            disp = np.linalg.norm(coords - x0_np, axis=1)
            return coords, disp

        def update_visualization(step):
            nonlocal cloud_actor
            coords, disp = get_step_data(step)

            cloud = pv.PolyData(coords)
            cloud["Displacement"] = disp

            if cloud_actor:
                plotter.remove_actor(cloud_actor)

            cloud_actor = plotter.add_mesh(
                cloud, scalars="Displacement", cmap="viridis",
                render_points_as_spheres=True, point_size=6,
                show_scalar_bar=True, clim=[np.min(disp), np.max(disp)]
            )
            plotter.update_scalar_bar_range([np.min(disp), np.max(disp)])
            plotter.add_text("Displacement", font_size=10, position='lower_left')
            plotter.render()
            if recording["enabled"]:
                gif_frames.append(plotter.screenshot(transparent_background=False))

        def slider_callback(value):
            step = int(round(value))
            update_visualization(step)

        def toggle_recording(value):
            recording["enabled"] = value
            print("Recording:", "ON" if value else "OFF")

        plotter.add_mesh(stl_mesh, color='gray', opacity=0.2)
        plotter.add_slider_widget(slider_callback, rng=[0, len(self.positions_over_time) - 1], value=0, title="Time Step")

        plotter.add_checkbox_button_widget(toggle_recording, value=False, position=(10, 80), size=20, color_on="yellow")
        plotter.add_text("Record GIF", position=(40, 80), font_size=10)

        update_visualization(idx)
        plotter.show()

        if gif_frames:
            imageio.mimsave(gif_path, gif_frames, duration=0.5)
            print(f"[{datetime.now().strftime('%I:%M:%S %p')}] Saved GIF: {gif_path}")


    def _visualize_displacement_vectors(self):    
        disp = (self.x - self.x0)
        x0_np = self.x0

        plt.figure(figsize=(6,6))
        plt.quiver(x0_np[:,0], x0_np[:,1], disp[:,0], disp[:,1], scale=0.5, width=0.002, color='black')
        plt.title("Displacement Vectors in XY Plane")
        plt.axis('equal')
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    from config_io import load_config
    config = load_config("config/model_config.yaml")         # or config.json 
    ID = config.get("id_output", "")
    PART_NAME = os.path.basename(config['part_STL']).split(".")[0] if 'part_STL' in config else "implicit_mpm_sim"
    MODEL_RESULTS = f"results/{PART_NAME}_sim{ID}.npz" if PART_NAME else "results/implicit_mpm_sim.npz"
    NODE_INPUT = f"geometries/pointcloud/{PART_NAME}_{config['nodes_per_layer']}NPL_{config['layer_height']}LH_nodes.npz"  # contains 'nodes_part', and your part_config blob if used
    STL_PATH = f"geometries/stl/{PART_NAME}.stl" if PART_NAME else "geometries/stl/implicit_mpm_sim.stl"


    plot = AM_Visualizer(MODEL_RESULTS, config)