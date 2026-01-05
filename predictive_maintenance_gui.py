import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.stats import kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import joblib
import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import ttk
from tkinter import messagebox
import tkinter as tk
from PIL import Image, ImageTk


# -----------------------------
# File info dictionary
# -----------------------------
# Περιέχει όλα τα αρχεία με τις βασικές τους ιδιότητες
FILE_INFO = {
    # ================= NORMAL =================
    "Normal_0.mat": {"Type": "Normal", "Motor Speed": 1797, "Motor Load": 0, "Fault Diameter": None},
    "Normal_1.mat": {"Type": "Normal", "Motor Speed": 1772, "Motor Load": 1, "Fault Diameter": None},
    "Normal_2.mat": {"Type": "Normal", "Motor Speed": 1750, "Motor Load": 2, "Fault Diameter": None},
    "Normal_3.mat": {"Type": "Normal", "Motor Speed": 1730, "Motor Load": 3, "Fault Diameter": None},

    # ================= INNER RACE 0.007 =================
    "IR007_0.mat": {"Type": "InnerRace", "Motor Speed": 1797, "Motor Load": 0, "Fault Diameter": 0.007},
    "IR007_1.mat": {"Type": "InnerRace", "Motor Speed": 1772, "Motor Load": 1, "Fault Diameter": 0.007},
    "IR007_2.mat": {"Type": "InnerRace", "Motor Speed": 1750, "Motor Load": 2, "Fault Diameter": 0.007},
    "IR007_3.mat": {"Type": "InnerRace", "Motor Speed": 1730, "Motor Load": 3, "Fault Diameter": 0.007},
     
    # ================= INNER RACE 0.014 =================
    "IR014_0.mat": {"Type": "InnerRace", "Motor Speed": 1797, "Motor Load": 0, "Fault Diameter": 0.014},
    "IR014_1.mat": {"Type": "InnerRace", "Motor Speed": 1772, "Motor Load": 1, "Fault Diameter": 0.014},
    "IR014_2.mat": {"Type": "InnerRace", "Motor Speed": 1750, "Motor Load": 2, "Fault Diameter": 0.014},
    "IR014_3.mat": {"Type": "InnerRace", "Motor Speed": 1730, "Motor Load": 3, "Fault Diameter": 0.014},

    # ================= INNER RACE 0.021 =================
    "IR021_0.mat": {"Type": "InnerRace", "Motor Speed": 1797, "Motor Load": 0, "Fault Diameter": 0.021},
    "IR021_1.mat": {"Type": "InnerRace", "Motor Speed": 1772, "Motor Load": 1, "Fault Diameter": 0.021},
    "IR021_2.mat": {"Type": "InnerRace", "Motor Speed": 1750, "Motor Load": 2, "Fault Diameter": 0.021},
    "IR021_3.mat": {"Type": "InnerRace", "Motor Speed": 1730, "Motor Load": 3, "Fault Diameter": 0.021},

    # ================= INNER RACE 0.028 =================
    "IR028_0.mat": {"Type": "InnerRace", "Motor Speed": 1797, "Motor Load": 0, "Fault Diameter": 0.028},
    "IR028_1.mat": {"Type": "InnerRace", "Motor Speed": 1772, "Motor Load": 1, "Fault Diameter": 0.028},
    "IR028_2.mat": {"Type": "InnerRace", "Motor Speed": 1750, "Motor Load": 2, "Fault Diameter": 0.028},
    "IR028_3.mat": {"Type": "InnerRace", "Motor Speed": 1730, "Motor Load": 3, "Fault Diameter": 0.028},
    
    # ================= BALL 0.007 =================
    "B007_0.mat": {"Type": "Ball", "Motor Speed": 1797, "Motor Load": 0, "Fault Diameter": 0.007},
    "B007_1.mat": {"Type": "Ball", "Motor Speed": 1772, "Motor Load": 1, "Fault Diameter": 0.007},
    "B007_2.mat": {"Type": "Ball", "Motor Speed": 1750, "Motor Load": 2, "Fault Diameter": 0.007},
    "B007_3.mat": {"Type": "Ball", "Motor Speed": 1730, "Motor Load": 3, "Fault Diameter": 0.007},

    # ================= BALL 0.014 =================
    "B014_0.mat": {"Type": "Ball", "Motor Speed": 1797, "Motor Load": 0, "Fault Diameter": 0.014},
    "B014_1.mat": {"Type": "Ball", "Motor Speed": 1772, "Motor Load": 1, "Fault Diameter": 0.014},
    "B014_2.mat": {"Type": "Ball", "Motor Speed": 1750, "Motor Load": 2, "Fault Diameter": 0.014},
    "B014_3.mat": {"Type": "Ball", "Motor Speed": 1730, "Motor Load": 3, "Fault Diameter": 0.014},

    # ================= BALL 0.021 =================
    "B021_0.mat": {"Type": "Ball", "Motor Speed": 1797, "Motor Load": 0, "Fault Diameter": 0.021},
    "B021_1.mat": {"Type": "Ball", "Motor Speed": 1772, "Motor Load": 1, "Fault Diameter": 0.021},
    "B021_2.mat": {"Type": "Ball", "Motor Speed": 1750, "Motor Load": 2, "Fault Diameter": 0.021},
    "B021_3.mat": {"Type": "Ball", "Motor Speed": 1730, "Motor Load": 3, "Fault Diameter": 0.021},

    # ================= BALL 0.028 =================
    "B028_0.mat": {"Type": "Ball", "Motor Speed": 1797, "Motor Load": 0, "Fault Diameter": 0.028},
    "B028_1.mat": {"Type": "Ball", "Motor Speed": 1772, "Motor Load": 1, "Fault Diameter": 0.028},
    "B028_2.mat": {"Type": "Ball", "Motor Speed": 1750, "Motor Load": 2, "Fault Diameter": 0.028},
    "B028_3.mat": {"Type": "Ball", "Motor Speed": 1730, "Motor Load": 3, "Fault Diameter": 0.028},

    # ================= OUTER RACE 0.007 =================
    "OR007@6_0.mat": {"Type": "OuterRace", "Motor Speed": 1797, "Motor Load": 0, "Fault Diameter": 0.007, "Position": "6:00"},
    "OR007@3_0.mat": {"Type": "OuterRace", "Motor Speed": 1797, "Motor Load": 0, "Fault Diameter": 0.007, "Position": "3:00"},
    "OR007@12_0.mat": {"Type": "OuterRace", "Motor Speed": 1797, "Motor Load": 0, "Fault Diameter": 0.007, "Position": "12:00"},
    "OR007@6_1.mat": {"Type": "OuterRace", "Motor Speed": 1772, "Motor Load": 1, "Fault Diameter": 0.007, "Position": "6:00"},
    "OR007@3_1.mat": {"Type": "OuterRace", "Motor Speed": 1772, "Motor Load": 1, "Fault Diameter": 0.007, "Position": "3:00"},
    "OR007@12_1.mat": {"Type": "OuterRace", "Motor Speed": 1772, "Motor Load": 1, "Fault Diameter": 0.007, "Position": "12:00"},
    "OR007@6_2.mat": {"Type": "OuterRace", "Motor Speed": 1750, "Motor Load": 2, "Fault Diameter": 0.007, "Position": "6:00"},
    "OR007@3_2.mat": {"Type": "OuterRace", "Motor Speed": 1750, "Motor Load": 2, "Fault Diameter": 0.007, "Position": "3:00"},
    "OR007@12_2.mat": {"Type": "OuterRace", "Motor Speed": 1750, "Motor Load": 2, "Fault Diameter": 0.007, "Position": "12:00"},
    "OR007@6_3.mat": {"Type": "OuterRace", "Motor Speed": 1730, "Motor Load": 3, "Fault Diameter": 0.007, "Position": "6:00"},
    "OR007@3_3.mat": {"Type": "OuterRace", "Motor Speed": 1730, "Motor Load": 3, "Fault Diameter": 0.007, "Position": "3:00"},
    "OR007@12_3.mat": {"Type": "OuterRace", "Motor Speed": 1730, "Motor Load": 3, "Fault Diameter": 0.007, "Position": "12:00"},
    
    # ================= OUTER RACE 0.014 =================
    "OR014@6_0.mat": {"Type": "OuterRace", "Motor Speed": 1797, "Motor Load": 0, "Fault Diameter": 0.014, "Position": "6:00"},
    "OR014@3_0.mat": {"Type": "OuterRace", "Motor Speed": 1797, "Motor Load": 0, "Fault Diameter": 0.014, "Position": "3:00"},
    "OR014@12_0.mat": {"Type": "OuterRace", "Motor Speed": 1797, "Motor Load": 0, "Fault Diameter": 0.014, "Position": "12:00"},
    "OR014@6_1.mat": {"Type": "OuterRace", "Motor Speed": 1772, "Motor Load": 1, "Fault Diameter": 0.014, "Position": "6:00"},
    "OR014@3_1.mat": {"Type": "OuterRace", "Motor Speed": 1772, "Motor Load": 1, "Fault Diameter": 0.014, "Position": "3:00"},
    "OR014@12_1.mat": {"Type": "OuterRace", "Motor Speed": 1772, "Motor Load": 1, "Fault Diameter": 0.014, "Position": "12:00"},
    "OR014@6_2.mat": {"Type": "OuterRace", "Motor Speed": 1750, "Motor Load": 2, "Fault Diameter": 0.014, "Position": "6:00"},
    "OR014@3_2.mat": {"Type": "OuterRace", "Motor Speed": 1750, "Motor Load": 2, "Fault Diameter": 0.014, "Position": "3:00"},
    "OR014@12_2.mat": {"Type": "OuterRace", "Motor Speed": 1750, "Motor Load": 2, "Fault Diameter": 0.014, "Position": "12:00"},
    "OR014@6_3.mat": {"Type": "OuterRace", "Motor Speed": 1730, "Motor Load": 3, "Fault Diameter": 0.014, "Position": "6:00"},
    "OR014@3_3.mat": {"Type": "OuterRace", "Motor Speed": 1730, "Motor Load": 3, "Fault Diameter": 0.014, "Position": "3:00"},
    "OR014@12_3.mat": {"Type": "OuterRace", "Motor Speed": 1730, "Motor Load": 3, "Fault Diameter": 0.014, "Position": "12:00"},   
 
    # ================= OUTER RACE 0.021 =================
    "OR021@6_0.mat": {"Type": "OuterRace", "Motor Speed": 1797, "Motor Load": 0, "Fault Diameter": 0.021, "Position": "6:00"},
    "OR021@3_0.mat": {"Type": "OuterRace", "Motor Speed": 1797, "Motor Load": 0, "Fault Diameter": 0.021, "Position": "3:00"},
    "OR021@12_0.mat": {"Type": "OuterRace", "Motor Speed": 1797, "Motor Load": 0, "Fault Diameter": 0.021, "Position": "12:00"},
    "OR021@6_1.mat": {"Type": "OuterRace", "Motor Speed": 1772, "Motor Load": 1, "Fault Diameter": 0.021, "Position": "6:00"},
    "OR021@3_1.mat": {"Type": "OuterRace", "Motor Speed": 1772, "Motor Load": 1, "Fault Diameter": 0.021, "Position": "3:00"},
    "OR021@12_1.mat": {"Type": "OuterRace", "Motor Speed": 1772, "Motor Load": 1, "Fault Diameter": 0.021, "Position": "12:00"},
    "OR021@6_2.mat": {"Type": "OuterRace", "Motor Speed": 1750, "Motor Load": 2, "Fault Diameter": 0.021, "Position": "6:00"},
    "OR021@3_2.mat": {"Type": "OuterRace", "Motor Speed": 1750, "Motor Load": 2, "Fault Diameter": 0.021, "Position": "3:00"},
    "OR021@12_2.mat": {"Type": "OuterRace", "Motor Speed": 1750, "Motor Load": 2, "Fault Diameter": 0.021, "Position": "12:00"},
    "OR021@6_3.mat": {"Type": "OuterRace", "Motor Speed": 1730, "Motor Load": 3, "Fault Diameter": 0.021, "Position": "6:00"},
    "OR021@3_3.mat": {"Type": "OuterRace", "Motor Speed": 1730, "Motor Load": 3, "Fault Diameter": 0.021, "Position": "3:00"},
    "OR021@12_3.mat": {"Type": "OuterRace", "Motor Speed": 1730, "Motor Load": 3, "Fault Diameter": 0.021, "Position": "12:00"},   

    # ================= OUTER RACE 0.028 =================
    "OR028@6_0.mat": {"Type": "OuterRace", "Motor Speed": 1797, "Motor Load": 0, "Fault Diameter": 0.028, "Position": "6:00"},
    "OR028@3_0.mat": {"Type": "OuterRace", "Motor Speed": 1797, "Motor Load": 0, "Fault Diameter": 0.028, "Position": "3:00"},
    "OR028@12_0.mat": {"Type": "OuterRace", "Motor Speed": 1797, "Motor Load": 0, "Fault Diameter": 0.028, "Position": "12:00"},
    "OR028@6_1.mat": {"Type": "OuterRace", "Motor Speed": 1772, "Motor Load": 1, "Fault Diameter": 0.028, "Position": "6:00"},
    "OR028@3_1.mat": {"Type": "OuterRace", "Motor Speed": 1772, "Motor Load": 1, "Fault Diameter": 0.028, "Position": "3:00"},
    "OR028@12_1.mat": {"Type": "OuterRace", "Motor Speed": 1772, "Motor Load": 1, "Fault Diameter": 0.028, "Position": "12:00"},
    "OR028@6_2.mat": {"Type": "OuterRace", "Motor Speed": 1750, "Motor Load": 2, "Fault Diameter": 0.028, "Position": "6:00"},
    "OR028@3_2.mat": {"Type": "OuterRace", "Motor Speed": 1750, "Motor Load": 2, "Fault Diameter": 0.028, "Position": "3:00"},
    "OR028@12_2.mat": {"Type": "OuterRace", "Motor Speed": 1750, "Motor Load": 2, "Fault Diameter": 0.028, "Position": "12:00"},
    "OR028@6_3.mat": {"Type": "OuterRace", "Motor Speed": 1730, "Motor Load": 3, "Fault Diameter": 0.028, "Position": "6:00"},
    "OR028@3_3.mat": {"Type": "OuterRace", "Motor Speed": 1730, "Motor Load": 3, "Fault Diameter": 0.028, "Position": "3:00"},
    "OR028@12_3.mat": {"Type": "OuterRace", "Motor Speed": 1730, "Motor Load": 3, "Fault Diameter": 0.028, "Position": "12:00"},   


}


# -----------------------------
# Helper functions
# -----------------------------
def load_de_signal(mat_file):
    """Φορτώνει το Drive End σήμα από .mat αρχείο"""
    data = scipy.io.loadmat(mat_file)
    de_key = None
    for key in data.keys():
        if "DE_time" in key or "DE" in key:
            de_key = key
            break
    if de_key is None:
        raise ValueError(f"DE signal not found in {mat_file}")
    return data[de_key].flatten()

def extract_features(signal):
    """Υπολογίζει βασικά χαρακτηριστικά σήματος"""
    rms = np.sqrt(np.mean(signal**2))
    std_val = np.std(signal)
    peak2peak = np.ptp(signal)
    kurt = kurtosis(signal)
    crest = np.max(np.abs(signal)) / rms
    return {"RMS": rms, "STD": std_val, "Peak2Peak": peak2peak, "Kurtosis": kurt, "CrestFactor": crest}

def plot_time_domain(ax, signal, fs, title="Time Domain"):
    """Σχεδιάζει Time Domain plot με auto-scaling"""
    t = np.arange(len(signal)) / fs
    ax.plot(t[:int(fs*0.1)], signal[:int(fs*0.1)])
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Acceleration [g]")
    ax.set_title(title)
    ax.grid(True)
    ax.autoscale(enable=True, axis='y', tight=True)

def plot_fft(ax, signal, fs, title="FFT Spectrum"):
    """Σχεδιάζει FFT με auto-scaling"""
    N = len(signal)
    yf = fft(signal)
    xf = fftfreq(N, 1/fs)
    xf = xf[:N//2]
    yf = np.abs(yf[:N//2])
    ax.plot(xf, yf)
    ax.set_xlim(0, 5000)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.grid(True)
    ax.autoscale(enable=True, axis='y', tight=True)

# -----------------------------
# Splash Screen (GLOBAL)
# -----------------------------
def show_splash(root, duration=1000):
    splash = tk.Toplevel(root)
    splash.overrideredirect(True)

    # Φόρτωση εικόνας
    img = Image.open("splash.png")
    photo = ImageTk.PhotoImage(img)
    splash.photo = photo  # κρατάμε αναφορά
    label = tk.Label(splash, image=photo)
    label.pack()

    # Κεντράρισμα
    w, h = img.width, img.height
    x = (splash.winfo_screenwidth() // 2) - (w // 2)
    y = (splash.winfo_screenheight() // 2) - (h // 2)
    splash.geometry(f"{w}x{h}+{x}+{y}")

    # Κλείσιμο splash και εμφάνιση root
    def close_splash():
        splash.destroy()
        root.deiconify()  # εμφανίζουμε το κύριο παράθυρο μετά το splash

    splash.after(duration, close_splash)






# -----------------------------
# GUI Application
# -----------------------------
class VibrationApp:
    def __init__(self, master):
        self.master = master
        master.title("Predictive Maintenance - Vibration Analysis")
        master.protocol("WM_DELETE_WINDOW", self.exit_app)
        self.fs = 12000
        self.training_files = []
        self.model = None
        self.training_figures = []
        self.pred_figures = []
        self.mat_reader_figures = []

        # -----------------------------
        # Tabs
        # -----------------------------
        self.tab_control = ttk.Notebook(master)
        self.tab_train = ttk.Frame(self.tab_control)
        self.tab_predict = ttk.Frame(self.tab_control)
        self.tab_mat_reader = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tab_train, text="Training")
        self.tab_control.add(self.tab_predict, text="Prediction")
        self.tab_control.add(self.tab_mat_reader, text="MAT Reader")
        self.tab_control.pack(expand=1, fill='both')

        # -----------------------------
        # --- Training Tab ---
        # -----------------------------
        self.train_btn_frame = tk.Frame(self.tab_train)
        self.train_btn_frame.pack(pady=5)
        self.select_button = tk.Button(self.train_btn_frame, text="Select Training Files", command=self.load_training_files)
        self.select_button.pack(side=tk.LEFT, padx=5)
        self.train_button = tk.Button(self.train_btn_frame, text="Train Model", command=self.train_model, state='disabled')
        self.train_button.pack(side=tk.LEFT, padx=5)
        self.save_train_button = tk.Button(self.train_btn_frame, text="Save Training Plots", command=self.save_training_plots, state='disabled')
        self.save_train_button.pack(side=tk.LEFT, padx=5)
        self.copy_train_button = tk.Button(self.train_btn_frame, text="Copy All Info", command=self.copy_train_text)
        self.copy_train_button.pack(side=tk.LEFT, padx=5)
        self.exit_button = tk.Button(self.train_btn_frame, text="Exit", command=self.exit_app)
        self.exit_button.pack(side=tk.LEFT, padx=5)

        self.text_frame_train = tk.Frame(self.tab_train)
        self.text_frame_train.pack(fill=tk.BOTH, expand=True)
        self.scrollbar_train = tk.Scrollbar(self.text_frame_train)
        self.scrollbar_train.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_box_train = tk.Text(self.text_frame_train, wrap='word', yscrollcommand=self.scrollbar_train.set, height=15)
        self.text_box_train.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar_train.config(command=self.text_box_train.yview)

        self.plot_canvas_frame_train = tk.Frame(self.tab_train)
        self.plot_canvas_frame_train.pack(fill=tk.BOTH, expand=True)
        self.canvas_train = tk.Canvas(self.plot_canvas_frame_train)
        self.canvas_train.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.plot_scrollbar_train = tk.Scrollbar(self.plot_canvas_frame_train, orient="vertical", command=self.canvas_train.yview)
        self.plot_scrollbar_train.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas_train.configure(yscrollcommand=self.plot_scrollbar_train.set)
        self.inner_frame_train = tk.Frame(self.canvas_train)
        self.canvas_train.create_window((0,0), window=self.inner_frame_train, anchor=tk.NW)
        self.inner_frame_train.bind("<Configure>", lambda event: self.canvas_train.configure(scrollregion=self.canvas_train.bbox("all")))

        # -----------------------------
        # --- Prediction Tab ---
        # -----------------------------
        self.predict_btn_frame = tk.Frame(self.tab_predict)
        self.predict_btn_frame.pack(pady=5)
        self.predict_button = tk.Button(self.predict_btn_frame, text="Predict New File", command=self.predict_new_file, state='disabled')
        self.predict_button.pack(side=tk.LEFT, padx=5)
        self.save_pred_button = tk.Button(self.predict_btn_frame, text="Save Prediction Plots", command=self.save_prediction_plots, state='disabled')
        self.save_pred_button.pack(side=tk.LEFT, padx=5)
        self.copy_pred_button = tk.Button(self.predict_btn_frame, text="Copy All Info", command=self.copy_pred_text)
        self.copy_pred_button.pack(side=tk.LEFT, padx=5)
        self.exit_button_pred = tk.Button(self.predict_btn_frame, text="Exit", command=self.exit_app)
        self.exit_button_pred.pack(side=tk.LEFT, padx=5)

        self.text_frame_pred = tk.Frame(self.tab_predict)
        self.text_frame_pred.pack(fill=tk.BOTH, expand=True)
        self.scrollbar_pred = tk.Scrollbar(self.text_frame_pred)
        self.scrollbar_pred.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_box_pred = tk.Text(self.text_frame_pred, wrap='word', yscrollcommand=self.scrollbar_pred.set, height=20)
        self.text_box_pred.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar_pred.config(command=self.text_box_pred.yview)

        self.plot_canvas_frame_pred = tk.Frame(self.tab_predict)
        self.plot_canvas_frame_pred.pack(fill=tk.BOTH, expand=True)
        self.canvas_pred = tk.Canvas(self.plot_canvas_frame_pred)
        self.canvas_pred.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.plot_scrollbar_pred = tk.Scrollbar(self.plot_canvas_frame_pred, orient="vertical", command=self.canvas_pred.yview)
        self.plot_scrollbar_pred.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas_pred.configure(yscrollcommand=self.plot_scrollbar_pred.set)
        self.inner_frame_pred = tk.Frame(self.canvas_pred)
        self.canvas_pred.create_window((0,0), window=self.inner_frame_pred, anchor=tk.NW)
        self.inner_frame_pred.bind("<Configure>", lambda event: self.canvas_pred.configure(scrollregion=self.canvas_pred.bbox("all")))

        # -----------------------------
        # --- MAT Reader Tab ---
        # -----------------------------
        self.reader_btn_frame = tk.Frame(self.tab_mat_reader)
        self.reader_btn_frame.pack(pady=5)
        self.load_mat_button = tk.Button(self.reader_btn_frame, text="Load .MAT File", command=self.read_mat_file)
        self.load_mat_button.pack(side=tk.LEFT, padx=5)
        self.copy_mat_button = tk.Button(self.reader_btn_frame, text="Copy All Info", command=self.copy_mat_text)
        self.copy_mat_button.pack(side=tk.LEFT, padx=5)
        self.exit_button_mat = tk.Button(self.reader_btn_frame, text="Exit", command=self.exit_app)
        self.exit_button_mat.pack(side=tk.LEFT, padx=5)

        self.text_frame_mat = tk.Frame(self.tab_mat_reader)
        self.text_frame_mat.pack(fill=tk.BOTH, expand=True)
        self.scrollbar_mat = tk.Scrollbar(self.text_frame_mat)
        self.scrollbar_mat.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_box_mat = tk.Text(self.text_frame_mat, wrap='word', yscrollcommand=self.scrollbar_mat.set, height=20)
        self.text_box_mat.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar_mat.config(command=self.text_box_mat.yview)

        self.plot_canvas_frame_mat = tk.Frame(self.tab_mat_reader)
        self.plot_canvas_frame_mat.pack(fill=tk.BOTH, expand=True)
        self.canvas_mat = tk.Canvas(self.plot_canvas_frame_mat)
        self.canvas_mat.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.plot_scrollbar_mat = tk.Scrollbar(self.plot_canvas_frame_mat, orient="vertical", command=self.canvas_mat.yview)
        self.plot_scrollbar_mat.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas_mat.configure(yscrollcommand=self.plot_scrollbar_mat.set)
        self.inner_frame_mat = tk.Frame(self.canvas_mat)
        self.canvas_mat.create_window((0,0), window=self.inner_frame_mat, anchor=tk.NW)
        self.inner_frame_mat.bind("<Configure>", lambda event: self.canvas_mat.configure(scrollregion=self.canvas_mat.bbox("all")))

        # -----------------------------
        # Αν υπάρχει ήδη αποθηκευμένο μοντέλο, φορτώνουμε
        # -----------------------------
        if os.path.exists("rf_model.joblib"):
            try:
                self.model = joblib.load("rf_model.joblib")
                self.predict_button.config(state='normal')
                self.save_pred_button.config(state='normal')
                self.text_box_train.insert(tk.END, "Existing model loaded from rf_model.joblib\n")
            except Exception as e:
                self.text_box_train.insert(tk.END, f"Failed to load existing model: {e}\n")

    # -----------------------------
    # Training functions
    # -----------------------------
    def load_training_files(self):
        files = filedialog.askopenfilenames(title="Select Training .mat Files", filetypes=[("MAT files","*.mat")])
        if files:
            # Ταξινομεί πρώτα τα Normal
            files_sorted = sorted(files, key=lambda x: 0 if "Normal" in os.path.basename(x) else 1)
            self.training_files = list(files_sorted)
            self.text_box_train.insert(tk.END, f"Selected {len(files_sorted)} training files.\n")
            self.train_button.config(state='normal')

    def train_model(self):
        X = []
        y = []
        self.training_figures = []
        for widget in self.inner_frame_train.winfo_children():
            widget.destroy()
        self.text_box_train.delete(1.0, tk.END)

        for file in self.training_files:
            signal = load_de_signal(file)
            features = extract_features(signal)
            X.append(list(features.values()))
            filename = os.path.basename(file)
            # Αν υπάρχει στο FILE_INFO, παίρνουμε το Type
            if filename in FILE_INFO:
                y.append(FILE_INFO[filename]["Type"])
                # Εμφανίζουμε και τις επιπλέον πληροφορίες στο training tab
                info = FILE_INFO[filename]
                self.text_box_train.insert(tk.END, f"{filename} -> Type: {info['Type']}, Motor Speed: {info['Motor Speed']} rpm, Motor Load: {info['Motor Load']} HP, Fault Diameter: {info['Fault Diameter']} inches, Position: {info.get('Position','N/A')}\n")
            else:
                y.append("Unknown")
                self.text_box_train.insert(tk.END, f"{filename} -> Type: Unknown\n")

            fig, axes = plt.subplots(2,1, figsize=(6,4))
            plot_time_domain(axes[0], signal, self.fs, title=f"{filename} - Time Domain")
            plot_fft(axes[1], signal, self.fs, title=f"{filename} - FFT")
            plt.tight_layout()
            self.training_figures.append(fig)
            canvas = FigureCanvasTkAgg(fig, master=self.inner_frame_train)
            canvas.draw()
            canvas.get_tk_widget().pack(pady=5)

        # Εκπαίδευση μοντέλου
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(np.array(X), np.array(y))
        joblib.dump(self.model, "rf_model.joblib")

        # Safe cross-validation
        class_counts = [list(y).count(c) for c in set(y)]
        min_class_count = min(class_counts) if class_counts else 0
        if min_class_count >= 2:
            cv_folds = min(5, min_class_count)
            scores = cross_val_score(self.model, np.array(X), np.array(y), cv=cv_folds)
            self.text_box_train.insert(tk.END, f"- Training Accuracy ({cv_folds}-fold CV): {np.mean(scores):.2f}\n")
        else:
            self.text_box_train.insert(tk.END, "- Not enough samples per class for cross-validation\n")

        # Εμφάνιση πληροφοριών μοντέλου
        self.text_box_train.insert(tk.END, "Model trained and saved as rf_model.joblib\n")
        self.text_box_train.insert(tk.END, "Model Info:\n")
        self.text_box_train.insert(tk.END, f"- Type: {type(self.model).__name__}\n")
        self.text_box_train.insert(tk.END, f"- Number of trees: {self.model.n_estimators}\n")
        feature_names = ["RMS", "STD", "Peak2Peak", "Kurtosis", "CrestFactor"]
        self.text_box_train.insert(tk.END, "- Feature Importances:\n")
        for feat, imp in zip(feature_names, self.model.feature_importances_):
            self.text_box_train.insert(tk.END, f"    {feat}: {imp:.3f}\n")

        self.save_train_button.config(state='normal')
        self.predict_button.config(state='normal')
        self.master.update_idletasks()

    # -----------------------------
    # Copy functions
    # -----------------------------
    def copy_train_text(self):
        text = self.text_box_train.get(1.0, tk.END)
        self.master.clipboard_clear()
        self.master.clipboard_append(text)
        messagebox.showinfo("Copy", "Training info copied to clipboard!")

    def copy_pred_text(self):
        text = self.text_box_pred.get(1.0, tk.END)
        self.master.clipboard_clear()
        self.master.clipboard_append(text)
        messagebox.showinfo("Copy", "Prediction info copied to clipboard!")

    def copy_mat_text(self):
        text = self.text_box_mat.get(1.0, tk.END)
        self.master.clipboard_clear()
        self.master.clipboard_append(text)
        messagebox.showinfo("Copy", "MAT info copied to clipboard!")

    # -----------------------------
    # Prediction functions
    # -----------------------------
    def predict_new_file(self):
        file = filedialog.askopenfilename(title="Select New .mat File", filetypes=[("MAT files","*.mat")])
        if not file:
            return
        signal = load_de_signal(file)
        features = extract_features(signal)
        X_new = np.array([list(features.values())])
        if self.model is None:
            self.model = joblib.load("rf_model.joblib")

        # Info μοντέλου
        self.text_box_pred.delete(1.0, tk.END)
        self.text_box_pred.insert(tk.END, "Model Info (used for prediction):\n")
        self.text_box_pred.insert(tk.END, f"- Type: {type(self.model).__name__}\n")
        self.text_box_pred.insert(tk.END, f"- Number of trees: {self.model.n_estimators}\n")
        feature_names = ["RMS", "STD", "Peak2Peak", "Kurtosis", "CrestFactor"]
        self.text_box_pred.insert(tk.END, "- Feature Importances:\n")
        for feat, imp in zip(feature_names, self.model.feature_importances_):
            self.text_box_pred.insert(tk.END, f"    {feat}: {imp:.3f}\n")
        self.text_box_pred.insert(tk.END, "\n")

        prediction = self.model.predict(X_new)[0]
        self.text_box_pred.insert(tk.END, f"Prediction for {os.path.basename(file)}: {prediction}\n")
        self.text_box_pred.insert(tk.END, "--- Features ---\n")
        for k, v in features.items():
            self.text_box_pred.insert(tk.END, f"{k}: {v:.4f}\n")

        # Πληροφορίες από FILE_INFO αν υπάρχουν
        filename = os.path.basename(file)
        if filename in FILE_INFO:
            info = FILE_INFO[filename]
            self.text_box_pred.insert(tk.END, "\n--- Additional Info ---\n")
            self.text_box_pred.insert(tk.END, f"Motor Speed (rpm): {info['Motor Speed']}\n")
            self.text_box_pred.insert(tk.END, f"Motor Load (HP): {info['Motor Load']}\n")
            self.text_box_pred.insert(tk.END, f"Fault Diameter (inches): {info['Fault Diameter']}\n")
            if "Position" in info:
                self.text_box_pred.insert(tk.END, f"Position: {info['Position']}\n")

        # Plots
        for widget in self.inner_frame_pred.winfo_children():
            widget.destroy()
        fig, axes = plt.subplots(2,1, figsize=(6,4))
        plot_time_domain(axes[0], signal, self.fs, title=f"{filename} - Time Domain")
        plot_fft(axes[1], signal, self.fs, title=f"{filename} - FFT")
        plt.tight_layout()
        self.pred_figures = [fig]

        canvas = FigureCanvasTkAgg(fig, master=self.inner_frame_pred)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=5)
        self.save_pred_button.config(state='normal')

    # -----------------------------
    # MAT Reader functions
    # -----------------------------
    def read_mat_file(self):
        file = filedialog.askopenfilename(title="Select .MAT File", filetypes=[("MAT files","*.mat")])
        if not file:
            return
        self.text_box_mat.delete(1.0, tk.END)
        for widget in self.inner_frame_mat.winfo_children():
            widget.destroy()
        try:
            data = scipy.io.loadmat(file)
            self.text_box_mat.insert(tk.END, f"File: {os.path.basename(file)}\n")
            for key, val in data.items():
                if key.startswith("__"):
                    continue
                self.text_box_mat.insert(tk.END, f"\nKey: {key}\nType: {type(val).__name__}\nShape: {val.shape}\n")
                if isinstance(val, np.ndarray) and np.issubdtype(val.dtype, np.number):
                    self.text_box_mat.insert(tk.END, f"Min: {np.min(val):.4f}, Max: {np.max(val):.4f}, Mean: {np.mean(val):.4f}, Std: {np.std(val):.4f}\n")
                    # Αν υπάρχει DE, δείχνουμε mini plot
                    if "DE" in key:
                        fig, axes = plt.subplots(2,1, figsize=(5,3))
                        plot_time_domain(axes[0], val.flatten(), self.fs, title=f"{key} - Time Domain")
                        plot_fft(axes[1], val.flatten(), self.fs, title=f"{key} - FFT")
                        plt.tight_layout()
                        self.mat_reader_figures.append(fig)
                        canvas = FigureCanvasTkAgg(fig, master=self.inner_frame_mat)
                        canvas.draw()
                        canvas.get_tk_widget().pack(pady=5)

        except Exception as e:
            self.text_box_mat.insert(tk.END, f"Error reading .mat file: {e}\n")

    # -----------------------------
    # Save plots
    # -----------------------------
    def save_training_plots(self):
        folder = filedialog.askdirectory(title="Select Folder to Save Training Plots")
        if not folder:
            return
        for i, fig in enumerate(self.training_figures):
            fig.savefig(os.path.join(folder, f"training_plot_{i+1}.png"))
        messagebox.showinfo("Save", f"{len(self.training_figures)} training plots saved!")

    def save_prediction_plots(self):
        folder = filedialog.askdirectory(title="Select Folder to Save Prediction Plots")
        if not folder:
            return
        for i, fig in enumerate(self.pred_figures):
            fig.savefig(os.path.join(folder, f"prediction_plot_{i+1}.png"))
        messagebox.showinfo("Save", f"{len(self.pred_figures)} prediction plots saved!")

    # -----------------------------
    # Exit
    # -----------------------------
    def exit_app(self):
        answer = messagebox.askyesno(
            "Exit Application",
            "Are you sure you want to exit the application?"
        )
        if answer:
            self.master.quit()
            self.master.destroy()
 
# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()  # κρύβουμε το κύριο παράθυρο προσωρινά
    show_splash(root, 1000)  # 1 sec splash
    
    root.title("Predictive Maintenance - Vibration Analysis")
    
    # -----------------------------
    # Set custom icon
    # -----------------------------
    icon = tk.PhotoImage(file="icon.png")  # ή "icon.ico" αν έχεις .ico αρχείο
    root.iconphoto(True, icon)
    
    app = VibrationApp(root)
    root.mainloop()
