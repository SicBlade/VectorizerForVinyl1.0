import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox, colorchooser, ttk
from PIL import Image, ImageTk, ImageEnhance
import numpy as np
import cv2
import os
from scipy.spatial import KDTree

# Modern Theme Colors
BG_DARK = "#1e1e1e"
BG_MED = "#2d2d2d"
ACCENT = "#007acc"
TEXT_LIGHT = "#e0e0e0"

class VinylProMaster:
    def __init__(self, master):
        self.master = master
        self.master.title("Vinyl Pro - Ultimate Edition")
        self.master.geometry("1200x900")
        self.master.configure(bg=BG_DARK)

        # State
        self.palette = {}
        self.img_array = None
        self.tk_img = None
        self.setup_ui()

    def setup_ui(self):
        # --- Sidebar ---
        self.sidebar = tk.Frame(self.master, width=300, bg=BG_MED, padx=15, pady=15)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(self.sidebar, text="VINYL PRO", font=('Arial', 16, 'bold'), bg=BG_MED, fg=ACCENT).pack(pady=(0, 20))

        tk.Button(self.sidebar, text="1. Load Image", command=self.open_image, bg=ACCENT, fg="white", relief=tk.FLAT, height=2, font='bold').pack(fill=tk.X, pady=5)
        tk.Button(self.sidebar, text="Auto-Detect Palette", command=self.auto_palette, bg="#444", fg=TEXT_LIGHT, relief=tk.FLAT).pack(fill=tk.X, pady=5)

        # --- Sliders ---
        self.create_label("Contrast (Color Sharpness)")
        self.contrast_scale = self.create_slider(1.0, 3.0, 1.2, 0.1)
        
        self.create_label("Brightness (Fix Darkening)")
        self.bright_scale = self.create_slider(0.5, 3.0, 1.5, 0.1)

        self.create_label("Path Smoothing")
        self.smooth_scale = self.create_slider(1, 100, 5, 1)

        self.create_label("Small Detail Filter")
        self.speckle_scale = self.create_slider(0, 200, 10, 1)

        # Features
        self.wire_var = tk.BooleanVar(value=False)
        tk.Checkbutton(self.sidebar, text="Preview Wireframe", variable=self.wire_var, command=self.update_preview, 
                       bg=BG_MED, fg=TEXT_LIGHT, selectcolor=BG_DARK, activebackground=BG_MED).pack(anchor="w", pady=15)

        # Reset & Export
        tk.Button(self.sidebar, text="Reset Defaults", command=self.reset_defaults, bg="#555", fg="white", relief=tk.FLAT).pack(fill=tk.X, pady=(20, 5))
        
        tk.Label(self.sidebar, text="", bg=BG_MED).pack(fill=tk.Y, expand=True) 
        tk.Button(self.sidebar, text="EXPORT SVG", command=self.vectorize, bg="#28a745", fg="white", font=('Arial', 12, 'bold'), relief=tk.FLAT, height=2).pack(fill=tk.X, pady=10)

        # --- Viewport ---
        self.view_frame = tk.Frame(self.master, bg=BG_DARK)
        self.view_frame.pack(side=tk.RIGHT, fill="both", expand=True)
        self.canvas = tk.Canvas(self.view_frame, bg=BG_DARK, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=20, pady=20)

    def create_label(self, text):
        tk.Label(self.sidebar, text=text, bg=BG_MED, fg=TEXT_LIGHT, font=('Arial', 10)).pack(anchor="w", pady=(10, 0))

    def create_slider(self, start, end, val, res):
        s = tk.Scale(self.sidebar, from_=start, to=end, orient=tk.HORIZONTAL, bg=BG_MED, fg=ACCENT, 
                     highlightthickness=0, troughcolor=BG_DARK, resolution=res, command=self.update_preview)
        s.set(val)
        s.pack(fill=tk.X)
        return s

    def reset_defaults(self):
        self.contrast_scale.set(1.2)
        self.bright_scale.set(1.5)
        self.smooth_scale.set(5)
        self.speckle_scale.set(10)
        self.update_preview()

    def open_image(self):
        path = filedialog.askopenfilename()
        if path:
            self.img_array = np.array(Image.open(path).convert("RGB"))
            self.auto_palette()
            self.update_preview()

    def auto_palette(self):
        if self.img_array is None: return
        # Denoise for cleaner palette picking
        small = cv2.resize(self.img_array, (200, 200))
        pixels = small.reshape(-1, 3)
        unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
        top_colors = unique_colors[np.argsort(-counts)[:15]]
        self.palette = {f"c{i}": tuple(rgb) for i, rgb in enumerate(top_colors)}
        self.update_preview()

    def get_processed_map(self):
        pil_img = Image.fromarray(self.img_array)
        enh_c = ImageEnhance.Contrast(pil_img).enhance(self.contrast_scale.get())
        enh_b = ImageEnhance.Brightness(enh_c).enhance(self.bright_scale.get())
        img = np.array(enh_b)
        
        p_vals = list(self.palette.values())
        tree = KDTree(p_vals)
        h, w = img.shape[:2]
        indices = tree.query(img.reshape(-1, 3))[1].reshape(h, w)
        return img, indices, p_vals

    def update_preview(self, _=None):
        if self.img_array is None: return
        img, indices, p_vals = self.get_processed_map()
        
        display_buf = np.zeros_like(img)
        for i, color in enumerate(p_vals):
            display_buf[indices == i] = color

        if self.wire_var.get():
            display_buf = cv2.Canny(display_buf, 50, 150)
            display_buf = cv2.cvtColor(display_buf, cv2.COLOR_GRAY2RGB)
            display_buf[np.where((display_buf == [255,255,255]).all(axis=2))] = [0, 255, 255] 

        prev_img = Image.fromarray(display_buf)
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw < 100: cw, ch = 800, 600
        
        prev_img.thumbnail((cw, ch))
        self.tk_img = ImageTk.PhotoImage(prev_img)
        self.canvas.delete("all")
        self.canvas.create_image(cw//2, ch//2, anchor="center", image=self.tk_img)

    def vectorize(self):
        if self.img_array is None: return
        filename = simpledialog.askstring("Export", "Enter filename:") or "vector_output"
        save_path = os.path.join(os.path.expanduser("~"), "Downloads", f"{filename}.svg")
        
        img, indices, p_vals = self.get_processed_map()
        h, w = img.shape[:2]
        
        def get_lum(rgb): return 0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2]
        sorted_keys = sorted(range(len(p_vals)), key=lambda i: get_lum(p_vals[i]), reverse=True)
        bg_idx = sorted_keys[0]

        svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
        
        for i in reversed(sorted_keys):
            if i == bg_idx: continue 
            
            mask = (indices == i).astype(np.uint8) * 255
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2,2), np.uint8))
            
            # RETR_CCOMP captures the holes inside d, o, b and the gold inside the pot
            contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                hex_c = '#%02x%02x%02x' % p_vals[i]
                svg.append(f'<g fill="{hex_c}" fill-rule="evenodd">')
                
                all_paths = []
                for cnt in contours:
                    if cv2.contourArea(cnt) < self.speckle_scale.get(): continue
                    epsilon = (self.smooth_scale.get() / 20000.0) * cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, epsilon, True)
                    d = "M " + " ".join(f"{p[0][0]},{p[0][1]}" for p in approx) + " Z"
                    all_paths.append(d)
                
                if all_paths:
                    svg.append(f'<path d="{" ".join(all_paths)}" />')
                svg.append('</g>')
        
        svg.append('</svg>')
        with open(save_path, "w") as f: f.write("\n".join(svg))
        messagebox.showinfo("Exported", f"Success! Saved to Downloads.")

if __name__ == "__main__":
    root = tk.Tk()
    app = VinylProMaster(root)
    root.mainloop()