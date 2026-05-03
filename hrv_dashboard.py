import sys
import os

# ================= FORCE QT SAFE ENV =================
os.environ["PYQTGRAPH_QT_LIB"] = "PyQt5"
os.environ["QT_API"] = "pyqt5"

# ================= CREATE QApplication FIRST =================
from PyQt5.QtWidgets import QApplication
app = QApplication(sys.argv)   # ✅ MUST BE FIRST

# ================= NOW SAFE IMPORTS =================
import numpy as np
import scipy.signal as signal
import pyqtgraph as pg
import neurokit2 as nk
import wfdb

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QScrollArea, QFrame
)
from PyQt5.QtCore import Qt, QTimer


# ================= SIMPLE UI CARD =================
class Card(QFrame):
    def __init__(self, title, widget=None):
        super().__init__()
        layout = QVBoxLayout()

        title_lbl = QLabel(title)
        title_lbl.setStyleSheet("font-weight:bold; font-size:14px;")
        layout.addWidget(title_lbl)

        if widget:
            layout.addWidget(widget)

        self.setLayout(layout)


# ================= MAIN DASHBOARD =================
class ECGBoard(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("ECG Dashboard - PyQt Stable Version")
        self.resize(1200, 800)

        self.fs = 250
        self.raw = None
        self.time = None

        self.timer = QTimer()
        self.timer.timeout.connect(self.live_update)
        self.ptr = 0

        self.init_ui()

    # ---------------- UI ----------------
    def init_ui(self):
        root = QWidget()
        main_layout = QHBoxLayout()
        root.setLayout(main_layout)
        self.setCentralWidget(root)

        # Sidebar
        side = QFrame()
        side.setFixedWidth(250)
        s_layout = QVBoxLayout()

        self.btn_upload = QPushButton("Upload ECG")
        self.btn_upload.clicked.connect(self.upload)
        s_layout.addWidget(self.btn_upload)

        self.btn_run = QPushButton("Run Analysis")
        self.btn_run.clicked.connect(self.analyze)
        self.btn_run.setEnabled(False)
        s_layout.addWidget(self.btn_run)

        self.label = QLabel("No file selected")
        s_layout.addWidget(self.label)

        side.setLayout(s_layout)
        main_layout.addWidget(side)

        # Main area
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)

        content = QWidget()
        self.vbox = QVBoxLayout()

        # Plot widget (IMPORTANT: created AFTER QApplication)
        self.plot = pg.PlotWidget()
        self.curve = self.plot.plot(pen='g')

        self.vbox.addWidget(Card("ECG Signal", self.plot))

        content.setLayout(self.vbox)
        self.scroll.setWidget(content)
        main_layout.addWidget(self.scroll)

    # ---------------- LOAD DATA ----------------
    def upload(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select ECG File")
        if not file:
            return

        self.label.setText(file)

        record = wfdb.rdsamp(file[:-4])
        self.raw = record[0][:, 0]
        self.fs = record[1]['fs']
        self.time = np.arange(len(self.raw)) / self.fs

        self.btn_run.setEnabled(True)

    # ---------------- ANALYSIS ----------------
    def analyze(self):
        b, a = signal.butter(
            3,
            [0.5/(self.fs/2), 40/(self.fs/2)],
            btype="band"
        )
        filtered = signal.filtfilt(b, a, self.raw)

        _, info = nk.ecg_peaks(filtered, sampling_rate=self.fs)

        self.curve.setData(self.time, filtered)

        self.ptr = 0
        self.timer.start(30)

    # ---------------- LIVE UPDATE ----------------
    def live_update(self):
        window = int(5 * self.fs)

        if self.ptr + window >= len(self.raw):
            self.ptr = 0

        x = self.time[self.ptr:self.ptr+window]
        y = self.raw[self.ptr:self.ptr+window]

        self.curve.setData(x, y)
        self.ptr += int(self.fs * 0.03)


# ================= RUN APP =================
if __name__ == "__main__":
    window = ECGBoard()
    window.show()
    sys.exit(app.exec_())