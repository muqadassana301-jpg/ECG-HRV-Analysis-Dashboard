import sys
import os
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, welch

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFileDialog, QListWidget
)

from PyQt5.QtCore import QTimer, Qt
import pyqtgraph as pg
import wfdb


# ================= FILTER =================
def bandpass(ecg, fs):
    nyq = 0.5 * fs
    b, a = butter(3, [0.5 / nyq, 35 / nyq], btype='band')
    return filtfilt(b, a, ecg)


# ================= R PEAKS =================
def detect_rpeaks(ecg, fs):
    peaks, _ = find_peaks(ecg, distance=int(0.25 * fs), prominence=np.std(ecg) * 0.6)
    return peaks


# ================= MAIN =================
class ECG_HRV(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("ICU ECG + HRV DASHBOARD")
        self.showMaximized()

        pg.setConfigOption('background', '#0b1220')
        pg.setConfigOption('foreground', 'w')

        # ================= DATA =================
        self.ecg = None
        self.fs = 360
        self.ptr = 0
        self.window = 2500

        self.index = []
        self.sdnn = []
        self.rmssd = []
        self.stress = []
        self.counter = 0

        # ================= UI =================
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout()
        central.setLayout(layout)

        # TITLE
        title = QLabel("ICU ECG + HRV ANALYSIS DASHBOARD")
        title.setStyleSheet("color:#00f5d4; font-size:18px; font-weight:bold;")
        layout.addWidget(title)

        # LOAD BUTTON
        self.btn = QPushButton("Load ECG Folder")
        self.btn.setStyleSheet("background:#00b4d8; color:black; font-weight:bold;")
        self.btn.clicked.connect(self.load_file)
        layout.addWidget(self.btn)

        # FILE LIST
        self.file_list = QListWidget()
        self.file_list.setMaximumHeight(70)
        self.file_list.setStyleSheet("background:#111827; color:white;")
        layout.addWidget(self.file_list)

        # ================= ECG =================
        self.ecg_plot = pg.PlotWidget(title="ECG Signal")
        self.ecg_curve = self.ecg_plot.plot(pen='#00f5d4')
        self.r_plot = pg.ScatterPlotItem(brush='r', size=6)
        self.ecg_plot.addItem(self.r_plot)
        self.ecg_plot.setMinimumHeight(250)
        layout.addWidget(self.ecg_plot)

        # ================= MIDDLE PLOTS =================
        mid = QHBoxLayout()

        self.rr_plot = pg.PlotWidget(title="RR Interval")
        self.freq_plot = pg.PlotWidget(title="Frequency HRV")
        self.poincare = pg.PlotWidget(title="Poincaré Plot")

        mid.addWidget(self.rr_plot)
        mid.addWidget(self.freq_plot)
        mid.addWidget(self.poincare)

        layout.addLayout(mid)

        # ================= ICU HILL =================
        self.hrv_plot = pg.PlotWidget(title="HRV TREND ")
        self.hrv_plot.addLegend()
        self.hrv_plot.setMinimumHeight(180)
        layout.addWidget(self.hrv_plot)

        # ================= COLOR LEGEND =================
        legend = QLabel("SDNN = PINK | RMSSD = PURPLE | STRESS = LIGHT RED")
        legend.setStyleSheet("color:white; background:#111827; padding:4px;")
        layout.addWidget(legend)

        # ================= 🔥 STRAIGHT STAT LINE (FIXED) =================
        self.stat_line = QLabel()
        self.stat_line.setStyleSheet("""
            background-color:#111827;
            color:white;
            font-size:14px;
            padding:10px;
            font-weight:bold;
        """)
        layout.addWidget(self.stat_line)

        # ================= TIMER =================
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(200)

    # ================= LOAD FILE =================
    def load_file(self):

        folder = QFileDialog.getExistingDirectory(self, "Select ECG Folder")
        if not folder:
            return

        files = os.listdir(folder)
        self.file_list.clear()
        self.file_list.addItems(files)

        hea = [f for f in files if f.endswith(".hea")]

        if hea:
            base = hea[0].replace(".hea", "")
            record = wfdb.rdrecord(os.path.join(folder, base))
            ecg = record.p_signal[:, 0]
            self.fs = record.fs
            self.ecg = bandpass(ecg[:8000], self.fs)

        else:
            dat = [f for f in files if f.endswith(".dat")]
            if dat:
                ecg = np.fromfile(os.path.join(folder, dat[0]), dtype=np.float32)
                self.ecg = bandpass(ecg[:8000], self.fs)

        self.ptr = 0

    # ================= UPDATE =================
    def update(self):

        if self.ecg is None:
            return

        seg = self.ecg[self.ptr:self.ptr + self.window]
        t = np.arange(len(seg)) / self.fs

        peaks = detect_rpeaks(seg, self.fs)

        self.ecg_curve.setData(t, seg)

        if len(peaks) > 1:
            self.r_plot.setData(peaks / self.fs, seg[peaks])

        rr = np.diff(peaks) / self.fs

        if len(rr) < 3:
            self.ptr += 20
            return

        # ================= HRV =================
        hr = 60 / np.mean(rr)
        sdnn = np.std(rr)
        rmssd = np.sqrt(np.mean(np.diff(rr)**2))
        stress = sdnn / (np.mean(rr) + 1e-6)

        # ================= RR =================
        self.rr_plot.clear()
        self.rr_plot.plot(rr, pen='y', symbol='o')

        # ================= FREQ =================
        f, pxx = welch(rr - np.mean(rr), fs=4)
        self.freq_plot.clear()
        self.freq_plot.plot(f, pxx, pen='m')

        # ================= POINCARE =================
        self.poincare.clear()
        self.poincare.plot(rr[:-1], rr[1:], pen=None, symbol='o', symbolBrush='r')

        # ================= STORE =================
        self.index.append(self.counter)
        self.sdnn.append(sdnn)
        self.rmssd.append(rmssd)
        self.stress.append(stress)
        self.counter += 1

        # ================= ICU HILL =================
        self.hrv_plot.clear()
        self.hrv_plot.addLegend()

        x = np.array(self.index)

        def add_hill(y, color, name):

            y = np.array(y)
            if len(x) == 0 or len(y) == 0:
                return

            n = min(len(x), len(y))
            x2 = x[:n]
            y2 = y[:n]

            y2 = y2 / (np.max(y2) + 1e-6)

            curve = pg.PlotCurveItem(x2, y2, pen=pg.mkPen(color, width=2), name=name)

            fill = pg.FillBetweenItem(
                curve,
                pg.PlotCurveItem(x2, np.zeros(n)),
                brush=pg.mkBrush(color)
            )

            fill.setOpacity(0.22)

            self.hrv_plot.addItem(curve)
            self.hrv_plot.addItem(fill)

        add_hill(self.sdnn, '#ff4d6d', "SDNN")
        add_hill(self.rmssd, '#c77dff', "RMSSD")
        add_hill(self.stress, '#ff85a1', "Stress")

        # ================= 🔥 STRAIGHT STAT LINE =================
        self.stat_line.setText(
            f"HR: {hr:.2f}   |   "
            f"SDNN: {sdnn:.4f}   |   "
            f"RMSSD: {rmssd:.4f}   |   "
            f"STRESS: {stress:.4f}"
        )

        self.ptr += 20


# ================= RUN =================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ECG_HRV()
    win.show()
    sys.exit(app.exec_())