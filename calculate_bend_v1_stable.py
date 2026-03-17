import tkinter as tk
from tkinter import filedialog, messagebox
import os
import json
import threading
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")  # Ensure Tk backend for GUI
import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
from Bio.PDB import PDBParser

##############################################################################
# 1. Utility Functions
##############################################################################
def load_pdb_coordinates(pdb_file):
    """
    Loads all atomic coordinates from a PDB using Bio.PDB.
    Returns (coords, atom_info) where:
      coords: np.ndarray of shape (N,3)
      atom_info: list of strings describing each atom
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pdb_structure", pdb_file)
    
    coords = []
    atom_info = []
    
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    coords.append(atom.get_coord())
                    info_str = f"{chain.id}_{residue.get_id()[1]}_{atom.name}"
                    atom_info.append(info_str)
    
    coords = np.array(coords)
    return coords, atom_info

def best_fit_line(points):
    """
    Given Nx3 points, returns (center, direction) of best-fit line.
    Uses SVD to find the principal axis.
    If <2 points, returns (None, None).
    """
    if len(points) < 2:
        return None, None
    center = np.mean(points, axis=0)
    shifted = points - center
    _, _, Vt = np.linalg.svd(shifted, full_matrices=False)
    direction = Vt[0]
    return center, direction

def angle_between_vectors(v1, v2):
    """
    Returns the angle in degrees between v1 and v2.
    """
    if np.linalg.norm(v1) < 1e-12 or np.linalg.norm(v2) < 1e-12:
        return None
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    dot = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
    return np.degrees(np.arccos(dot))

##############################################################################
# 2. Main GUI
##############################################################################
class LassoTwoSetsGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Bend calculator")

        # Data attributes for main PDB
        self.pdb_file = None
        self.log_file = None
        self.coords_3d = None
        self.atom_info = None
        self.coords_2d = None

        # Selections
        self.set1_indices = set()
        self.set2_indices = set()
        self.temp_indices = set()

        # Best-fit lines for main PDB
        self.line1_data = None  # (center, direction)
        self.line2_data = None
        self.line1_3d = None
        self.line2_3d = None
        self.lines_visible = True

        # Overlay structure data (just the spline)
        self.overlay_structure = None
        self.overlay_coords_3d = None  # For bounding-box
        self.overlay_spline_lines = []  # list of line artists
        self.overlay_spline_visible = True

        # 3D grid visibility toggle
        self.show_grid = True

        # --- Menubar ---
        menubar = tk.Menu(self)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open RNA trajectory", command=self.menu_open_pdb)
        filemenu.add_command(label="Open RNA model", command=self.menu_open_overlay)
        filemenu.add_command(label="Save Log", command=self.menu_save_log)
        filemenu.add_command(label="Load Log", command=self.menu_load_log)
        filemenu.add_command(label="Save Figure", command=self.menu_save_figure)  # new command
        filemenu.add_separator()
        filemenu.add_command(label="Apply Settings to Folder", command=self.apply_settings_to_folder)
        filemenu.add_separator()
        filemenu.add_command(label="Quit", command=self.on_quit)
        menubar.add_cascade(label="File", menu=filemenu)
        self.config(menu=menubar)

        # --- Figure and subplots ---
        self.fig = plt.Figure(figsize=(8, 4), dpi=100)
        self.ax2d = self.fig.add_subplot(1, 2, 1)
        self.ax3d = self.fig.add_subplot(1, 2, 2, projection='3d')

        # Fix to orthographic projection
        self.ax3d.set_proj_type('ortho')

        # Turn on grid (can be toggled)
        self.ax3d.grid(self.show_grid)

        self.ax2d.set_title("2D view")
        self.ax2d.set_xlabel("X")
        self.ax2d.set_ylabel("Y")
        
        self.ax3d.set_title("3D View")
        self.ax3d.set_xlabel("X")
        self.ax3d.set_ylabel("Y")
        self.ax3d.set_zlabel("Z")

        # Create initial scatter placeholders
        self.scatter2d = self.ax2d.scatter([], [], s=10, c='blue')
        self.scatter3d = self.ax3d.scatter([], [], [], s=10, c='blue')

        # Embed in Tk
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # LassoSelector for the 2D subplot
        self.lasso = LassoSelector(
            ax=self.ax2d,
            onselect=self.on_lasso_select,
            props={'color': 'orange', 'linewidth': 1, 'alpha': 0.8}
        )

        # --- Bottom control panel ---
        button_frame = tk.Frame(self)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X)

        tk.Button(button_frame, text="Confirm Set 1", command=self.confirm_set1)\
          .pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(button_frame, text="Confirm Set 2", command=self.confirm_set2)\
          .pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(button_frame, text="Compute Angle", command=self.compute_angle)\
          .pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(button_frame, text="Toggle Lines", command=self.toggle_lines)\
          .pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(button_frame, text="Reset Selections", command=self.reset_selections)\
          .pack(side=tk.LEFT, padx=5, pady=5)

        # Button to toggle the overlay spline
        tk.Button(button_frame, text="Toggle Overlay Spline", command=self.toggle_overlay_spline)\
          .pack(side=tk.LEFT, padx=5, pady=5)

        # Button to toggle the 3D grid
        tk.Button(button_frame, text="Toggle Grid", command=self.toggle_grid)\
          .pack(side=tk.LEFT, padx=5, pady=5)

        # Angle label
        self.angle_label = tk.Label(button_frame, text="Angle: N/A")
        self.angle_label.pack(side=tk.LEFT, padx=10, pady=5)

        tk.Button(button_frame, text="Quit", command=self.on_quit)\
          .pack(side=tk.RIGHT, padx=5, pady=5)

        self.canvas.draw()

    ###########################################################################
    # File Menu
    ###########################################################################
    def menu_open_pdb(self):
        """Load the main PDB file."""
        filename = filedialog.askopenfilename(filetypes=[("PDB Files", "*.pdb"), ("All Files", "*.*")])
        if filename:
            self.load_new_pdb(filename)

    def menu_open_overlay(self):
        """Load a second PDB file, building only the spline (no scatter)."""
        filename = filedialog.askopenfilename(filetypes=[("PDB Files", "*.pdb"), ("All Files", "*.*")])
        if filename:
            self.load_overlay_pdb(filename)

    def menu_save_log(self):
        """Save current selections/lines to a log file."""
        initial = self.log_file if self.log_file else "lasso_log.json"
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
            initialfile=initial
        )
        if filename:
            self.write_log(file_path=filename)

    def menu_load_log(self):
        """Load selections/lines from a log file."""
        filename = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")])
        if filename:
            self.load_log(file_path=filename)

    def menu_save_figure(self):
        """Save the current figure as a PNG or SVG file."""
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Files", "*.png"), ("SVG Files", "*.svg"), ("All Files", "*.*")]
        )
        if filename:
            try:
                self.fig.savefig(filename)
                print(f"Figure saved as '{filename}'")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save figure:\n{e}")

    def apply_settings_to_folder(self):
        """Applies the current selection sets to all PDBs in a chosen folder."""
        if self.pdb_file is None:
            messagebox.showerror("Error", "No PDB file is loaded. Please load a PDB first.")
            return
        if len(self.set1_indices) < 2 or len(self.set2_indices) < 2:
            messagebox.showerror("Error", "Both selections must have at least 2 points.")
            return

        folder = filedialog.askdirectory(title="Select Folder Containing PDB Files")
        if not folder:
            return

        # Process in a separate thread
        threading.Thread(target=self._process_folder, args=(folder,), daemon=True).start()

    def _process_folder(self, folder):
        # Create subfolder for logs
        log_folder = os.path.join(folder, "bend_analysis_log_files")
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)

        current_atom_count = self.coords_3d.shape[0]
        results = []

        for fname in os.listdir(folder):
            if fname.lower().endswith(".pdb"):
                full_path = os.path.join(folder, fname)
                try:
                    coords, _ = load_pdb_coordinates(full_path)
                except Exception as e:
                    print(f"Failed to load {fname}: {e}")
                    continue

                if coords.shape[0] != current_atom_count:
                    print(f"Skipping {fname}: mismatch in atom count.")
                    continue

                pts1 = coords[list(self.set1_indices)]
                pts2 = coords[list(self.set2_indices)]
                c1, d1 = best_fit_line(pts1)
                c2, d2 = best_fit_line(pts2)
                if c1 is None or c2 is None:
                    print(f"Skipping {fname}: not enough points for line fitting.")
                    continue

                angle_deg = angle_between_vectors(d1, d2)
                if angle_deg is None:
                    print(f"Skipping {fname}: could not compute angle.")
                    continue

                print(f"File: {fname} -> Angle: {angle_deg:.4f}°")
                results.append({"File Name": fname, "Angle (°)": round(angle_deg, 4)})

                # Write log
                log_data = {
                    "pdb_file": full_path,
                    "set1_indices": sorted(map(int, self.set1_indices)),
                    "set2_indices": sorted(map(int, self.set2_indices)),
                    "line1_center": c1.tolist() if c1 is not None else None,
                    "line1_direction": d1.tolist() if d1 is not None else None,
                    "line2_center": c2.tolist() if c2 is not None else None,
                    "line2_direction": d2.tolist() if d2 is not None else None,
                    "angle_degrees": float(angle_deg) if angle_deg is not None else None
                }
                log_filename = os.path.join(log_folder, self.get_log_filename(full_path))
                try:
                    with open(log_filename, "w") as f:
                        json.dump(log_data, f, indent=2)
                    print(f"Log written for {fname} at {log_filename}")
                except Exception as e:
                    print(f"Failed to write log for {fname}: {e}")

        if not results:
            self.after(0, lambda: messagebox.showinfo("Results", "No matching PDB files processed."))
            return

        # Once done, ask user for an Excel filename
        self.after(0, lambda: self._save_excel_results(results))

    def _save_excel_results(self, results):
        excel_file = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel Files", "*.xlsx"), ("All Files", "*.*")],
            title="Save Excel Results As"
        )
        if not excel_file:
            return

        df = pd.DataFrame(results)
        try:
            df.to_excel(excel_file, index=False)
            messagebox.showinfo("Success", f"Results saved to {excel_file}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save Excel file:\n{e}")

    ###########################################################################
    # Load new PDB and Update
    ###########################################################################
    def load_new_pdb(self, pdb_file):
        """Load a new main PDB, reset selections, update plots."""
        self.pdb_file = pdb_file
        self.log_file = self.get_log_filename(pdb_file)

        try:
            self.coords_3d, self.atom_info = load_pdb_coordinates(pdb_file)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load PDB file:\n{e}")
            return

        if self.coords_3d.size == 0:
            messagebox.showwarning("Warning", "No coordinates found in the PDB!")
            return

        self.coords_2d = self.coords_3d[:, :2]
        self.set1_indices.clear()
        self.set2_indices.clear()
        self.temp_indices.clear()
        self.line1_data = None
        self.line2_data = None

        # Remove old lines
        if self.line1_3d is not None:
            self.line1_3d.remove()
            self.line1_3d = None
        if self.line2_3d is not None:
            self.line2_3d.remove()
            self.line2_3d = None

        # Clear overlay
        self.overlay_structure = None
        self.overlay_coords_3d = None
        self.clear_overlay_spline()

        # Clear and re-draw 2D
        self.ax2d.cla()
        self.ax2d.set_title("2D view")
        self.ax2d.set_xlabel("X")
        self.ax2d.set_ylabel("Y")
        self.scatter2d = self.ax2d.scatter(self.coords_2d[:, 0],
                                           self.coords_2d[:, 1],
                                           s=10, c='blue')

        # Clear and re-draw 3D
        self.ax3d.cla()
        self.ax3d.set_title("3D View")
        self.ax3d.set_xlabel("X")
        self.ax3d.set_ylabel("Y")
        self.ax3d.set_zlabel("Z")
        # Always orthographic
        self.ax3d.set_proj_type('ortho')
        self.ax3d.grid(self.show_grid)

        self.scatter3d = self.ax3d.scatter(self.coords_3d[:, 0],
                                           self.coords_3d[:, 1],
                                           self.coords_3d[:, 2],
                                           s=10, c='blue')

        # Update bounding box
        self.update_3d_bounding_box()

        self.canvas.draw()
        print(f"Loaded {self.coords_3d.shape[0]} points from '{pdb_file}'")

        # If a log file exists, ask to load it
        if os.path.isfile(self.log_file):
            ans = messagebox.askyesno("Load Log?", f"Log file '{self.log_file}' found.\nLoad previous selections?")
            if ans:
                self.load_log()

    def load_overlay_pdb(self, pdb_file):
        """
        Loads a second PDB, building only a spline (no scatter).
        For nucleic acids, we look for 'P' in each residue (DNA/RNA backbone).
        For proteins, you could replace 'P' with 'CA'.
        """
        parser = PDBParser(QUIET=True)
        try:
            self.overlay_structure = parser.get_structure("overlay", pdb_file)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to parse overlay PDB:\n{e}")
            return

        # Grab all coords from the overlay structure for bounding-box
        coords_list = []
        for model in self.overlay_structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        coords_list.append(atom.coord)
        if coords_list:
            self.overlay_coords_3d = np.array(coords_list)
        else:
            self.overlay_coords_3d = None

        # Remove any old lines
        self.clear_overlay_spline()
        # Build new spline lines
        self.build_overlay_spline(backbone_atom='P')  # or 'CA' if protein

        # Update bounding box again (union of main + overlay)
        self.update_3d_bounding_box()

        self.canvas.draw()
        print(f"Overlay spline built from '{pdb_file}'.")

    ###########################################################################
    # Overlay Spline
    ###########################################################################
    def build_overlay_spline(self, backbone_atom='P'):
        """
        Build a simple 'spline' (thick line) by connecting backbone atoms 
        in each chain of the overlay structure. Typically 'P' for DNA/RNA, 'CA' for protein.
        """
        if not self.overlay_structure:
            return

        for chain in self.overlay_structure.get_chains():
            coords = []
            for residue in chain:
                if backbone_atom in residue:
                    coords.append(residue[backbone_atom].coord)
            if len(coords) > 1:
                coords = np.array(coords)
                # Draw a thick magenta line
                line, = self.ax3d.plot(coords[:, 0],
                                       coords[:, 1],
                                       coords[:, 2],
                                       color='magenta',
                                       linewidth=3,
                                       visible=self.overlay_spline_visible)
                self.overlay_spline_lines.append(line)

    def clear_overlay_spline(self):
        """Remove existing spline lines."""
        for ln in self.overlay_spline_lines:
            ln.remove()
        self.overlay_spline_lines = []

    def toggle_overlay_spline(self):
        """Toggle the visibility of the overlay spline lines."""
        self.overlay_spline_visible = not self.overlay_spline_visible
        for ln in self.overlay_spline_lines:
            ln.set_visible(self.overlay_spline_visible)
        self.canvas.draw()

    ###########################################################################
    # Force a cubic bounding box
    ###########################################################################
    def update_3d_bounding_box(self):
        """
        Sets the 3D axes so that x, y, z each span the same distance (a cube).
        This prevents distortion of the displayed molecule.
        """
        # Collect all relevant coords: main and overlay
        all_coords = []
        if self.coords_3d is not None and len(self.coords_3d) > 0:
            all_coords.append(self.coords_3d)
        if self.overlay_coords_3d is not None and len(self.overlay_coords_3d) > 0:
            all_coords.append(self.overlay_coords_3d)
        if not all_coords:
            return  # nothing to bound

        combined = np.vstack(all_coords)
        x_min, x_max = combined[:,0].min(), combined[:,0].max()
        y_min, y_max = combined[:,1].min(), combined[:,1].max()
        z_min, z_max = combined[:,2].min(), combined[:,2].max()

        dx = x_max - x_min
        dy = y_max - y_min
        dz = z_max - z_min
        max_range = max(dx, dy, dz)

        mid_x = 0.5 * (x_min + x_max)
        mid_y = 0.5 * (y_min + y_max)
        mid_z = 0.5 * (z_min + z_max)

        self.ax3d.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
        self.ax3d.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
        self.ax3d.set_zlim(mid_z - max_range/2, mid_z + max_range/2)

        # This ensures the box is truly cubic
        self.ax3d.set_box_aspect((1,1,1))
        # Keep grid consistent with current setting
        self.ax3d.grid(self.show_grid)

    ###########################################################################
    # Toggle Grid (Hide Axes/Background)
    ###########################################################################
    def toggle_grid(self):
        """
        Toggles the visibility of the 3D grid and axis lines.
        When toggled off, everything is hidden and background is white.
        When toggled on, normal axis lines/grid are restored.
        """
        self.show_grid = not self.show_grid

        if self.show_grid:
            # Turn the axis "on"
            self.ax3d._axis3don = True
            # Restore grid lines
            self.ax3d.grid(True)
        else:
            # Turn the axis "off" (hides bounding box & ticks in 3D)
            self.ax3d._axis3don = False
            # Also hide grid lines
            self.ax3d.grid(False)

        # Force background to white in both states
        self.ax3d.set_facecolor('white')

        self.canvas.draw()
        print(f"3D axes/grid toggled. show_grid={self.show_grid}")

    ###########################################################################
    # Lasso and Selections
    ###########################################################################
    def on_lasso_select(self, verts):
        if self.coords_2d is None or len(self.coords_2d) == 0:
            return
        path = Path(verts)
        contained = path.contains_points(self.coords_2d)
        self.temp_indices = set(np.where(contained)[0])
        self.update_colors()

    def confirm_set1(self):
        self.set1_indices.clear()
        self.set1_indices.update(self.temp_indices)
        print(f"Confirmed Set 1 with {len(self.set1_indices)} points.")
        self.update_colors()

    def confirm_set2(self):
        self.set2_indices.clear()
        self.set2_indices.update(self.temp_indices)
        print(f"Confirmed Set 2 with {len(self.set2_indices)} points.")
        self.update_colors()

    def update_colors(self):
        """Update color coding for the main PDB scatter."""
        if self.coords_3d is None:
            return
        color2d = []
        color3d = []
        for i in range(len(self.coords_3d)):
            if i in self.set1_indices:
                color2d.append('red')
                color3d.append('red')
            elif i in self.set2_indices:
                color2d.append('green')
                color3d.append('green')
            elif i in self.temp_indices:
                color2d.append('orange')
                color3d.append('orange')
            else:
                color2d.append('blue')
                color3d.append('blue')

        self.scatter2d.set_color(color2d)
        self.scatter3d._facecolor3d = color3d
        self.scatter3d._edgecolor3d = color3d
        self.canvas.draw()

    def reset_selections(self):
        """Clear all sets, remove lines, reset angle display."""
        self.set1_indices.clear()
        self.set2_indices.clear()
        self.temp_indices.clear()
        self.line1_data = None
        self.line2_data = None

        if self.line1_3d is not None:
            self.line1_3d.remove()
            self.line1_3d = None
        if self.line2_3d is not None:
            self.line2_3d.remove()
            self.line2_3d = None

        self.angle_label.config(text="Angle: N/A")
        self.update_colors()
        self.canvas.draw()
        print("Selections reset.")

    ###########################################################################
    # Compute Angle & Draw Lines
    ###########################################################################
    def compute_angle(self):
        """Compute angle between best-fit lines of Set1 and Set2."""
        if len(self.set1_indices) < 2 or len(self.set2_indices) < 2:
            print("Need >=2 points in each set to fit lines.")
            return

        pts1 = self.coords_3d[list(self.set1_indices)]
        pts2 = self.coords_3d[list(self.set2_indices)]

        c1, d1 = best_fit_line(pts1)
        c2, d2 = best_fit_line(pts2)
        if c1 is None or c2 is None:
            print("Error computing best-fit lines.")
            return

        self.line1_data = (c1, d1)
        self.line2_data = (c2, d2)

        angle_deg = angle_between_vectors(d1, d2)
        if angle_deg is None:
            print("Angle could not be computed (zero-length direction?).")
            return

        print(f"Angle between lines: {angle_deg:.4f}°")
        self.angle_label.config(text=f"Angle: {angle_deg:.4f}°")
        self.draw_lines()

    def draw_lines(self):
        """Draw best-fit lines for each set in 3D."""
        if self.line1_3d is not None:
            self.line1_3d.remove()
            self.line1_3d = None
        if self.line2_3d is not None:
            self.line2_3d.remove()
            self.line2_3d = None

        if not self.line1_data or not self.line2_data:
            return

        c1, d1 = self.line1_data
        c2, d2 = self.line2_data

        # Range for Set1
        pts1 = self.coords_3d[list(self.set1_indices)]
        t1_vals = np.dot(pts1 - c1, d1)
        t1_min, t1_max = t1_vals.min(), t1_vals.max()
        margin1 = (t1_max - t1_min) * 0.1 if t1_max != t1_min else 1
        t1_min -= margin1
        t1_max += margin1
        t1 = np.linspace(t1_min, t1_max, 50)
        line1_pts = c1[None, :] + t1[:, None] * d1[None, :]

        # Range for Set2
        pts2 = self.coords_3d[list(self.set2_indices)]
        t2_vals = np.dot(pts2 - c2, d2)
        t2_min, t2_max = t2_vals.min(), t2_vals.max()
        margin2 = (t2_max - t2_min) * 0.1 if t2_max != t2_min else 1
        t2_min -= margin2
        t2_max += margin2
        t2 = np.linspace(t2_min, t2_max, 50)
        line2_pts = c2[None, :] + t2[:, None] * d2[None, :]

        # Plot them
        self.line1_3d, = self.ax3d.plot(
            line1_pts[:, 0],
            line1_pts[:, 1],
            line1_pts[:, 2],
            color='red',
            linewidth=2,
            visible=self.lines_visible
        )
        self.line2_3d, = self.ax3d.plot(
            line2_pts[:, 0],
            line2_pts[:, 1],
            line2_pts[:, 2],
            color='green',
            linewidth=2,
            visible=self.lines_visible
        )
        self.canvas.draw()

    def toggle_lines(self):
        """Toggle visibility of best-fit lines."""
        self.lines_visible = not self.lines_visible
        if self.line1_3d is not None:
            self.line1_3d.set_visible(self.lines_visible)
        if self.line2_3d is not None:
            self.line2_3d.set_visible(self.lines_visible)
        self.canvas.draw()

    ###########################################################################
    # Logging
    ###########################################################################
    def write_log(self, file_path=None):
        """Write a JSON log with the current sets, lines, and angle."""
        if not self.line1_data or not self.line2_data:
            angle_deg = None
        else:
            _, d1 = self.line1_data
            _, d2 = self.line2_data
            angle_deg = angle_between_vectors(d1, d2)

        log_data = {
            "pdb_file": self.pdb_file,
            "set1_indices": sorted(map(int, self.set1_indices)),
            "set2_indices": sorted(map(int, self.set2_indices)),
            "line1_center": self.line1_data[0].tolist() if self.line1_data else None,
            "line1_direction": self.line1_data[1].tolist() if self.line1_data else None,
            "line2_center": self.line2_data[0].tolist() if self.line2_data else None,
            "line2_direction": self.line2_data[1].tolist() if self.line2_data else None,
            "angle_degrees": float(angle_deg) if angle_deg is not None else None
        }

        target_file = file_path if file_path else self.log_file
        if not target_file:
            messagebox.showerror("Error", "No log file specified.")
            return

        with open(target_file, "w") as f:
            json.dump(log_data, f, indent=2)
        print(f"Log written to '{target_file}'")
        self.log_file = target_file

    def load_log(self, file_path=None):
        """Read a JSON log and restore sets/lines."""
        target_file = file_path if file_path else self.log_file
        if not target_file or not os.path.isfile(target_file):
            print(f"No log file found at '{target_file}'")
            return

        with open(target_file, "r") as f:
            log_data = json.load(f)

        if log_data.get("pdb_file") != self.pdb_file:
            print("Warning: log file is for a different PDB.")

        self.set1_indices = set(log_data["set1_indices"])
        self.set2_indices = set(log_data["set2_indices"])

        line1_center = log_data.get("line1_center")
        line1_direction = log_data.get("line1_direction")
        line2_center = log_data.get("line2_center")
        line2_direction = log_data.get("line2_direction")

        if line1_center and line1_direction:
            c1 = np.array(line1_center)
            d1 = np.array(line1_direction)
            self.line1_data = (c1, d1)
        else:
            self.line1_data = None

        if line2_center and line2_direction:
            c2 = np.array(line2_center)
            d2 = np.array(line2_direction)
            self.line2_data = (c2, d2)
        else:
            self.line2_data = None

        self.temp_indices.clear()
        self.update_colors()
        self.draw_lines()

        angle = log_data.get("angle_degrees")
        if angle is not None:
            print(f"Log loaded. Angle was {angle:.4f}°.")
            self.angle_label.config(text=f"Angle: {angle:.4f}°")
        else:
            print("Log loaded. Angle not computed yet.")
        self.log_file = target_file

    ###########################################################################
    # Misc
    ###########################################################################
    def get_log_filename(self, pdb_file):
        """Generate a default log filename from the PDB filename."""
        base = os.path.basename(pdb_file)
        root, _ = os.path.splitext(base)
        return f"{root}_lasso_log.json"

    def on_quit(self):
        self.destroy()

##############################################################################
# 3. Run
##############################################################################
if __name__ == "__main__":
    app = LassoTwoSetsGUI()
    app.mainloop()
