import streamlit as st
import numpy as np
import wfdb
import zipfile
import os
from scipy.signal import butter, filtfilt, find_peaks, welch
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# ================= PAGE =================
st.set_page_config(layout="wide")

st.markdown("""
<style>
.stApp {background: #f5f7fb;}
.header {
    background: linear-gradient(90deg,#1e3a8a,#2563eb);
    padding: 18px;
    border-radius: 14px;
    font-size: 26px;
    text-align: center;
    color: white;
}
.card {
    background: white;
    padding: 16px;
    border-radius: 14px;
    box-shadow: 0px 3px 12px rgba(0,0,0,0.08);
    margin-top: 10px;
}
.warning {
    background: #fff3cd;
    padding: 10px;
    border-radius: 10px;
    color: #856404;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header">ECG & HRV Analysis Dashboard</div>', unsafe_allow_html=True)

# ================= SIDEBAR WARNING =================
st.markdown("""
<div class="warning">
⚠ Use sliders only if you understand signal processing (filtering affects ECG morphology and HRV results).
</div>
""", unsafe_allow_html=True)

# ================= SIDEBAR =================
st.sidebar.header("ECG Input")

mode = st.sidebar.radio("Select Mode", ["ZIP", "DAT + HEA", "Record"])

zip_file = dat_file = hea_file = record_name = None

if mode == "ZIP":
    zip_file = st.sidebar.file_uploader("Upload ZIP", type=["zip"])
elif mode == "DAT + HEA":
    dat_file = st.sidebar.file_uploader("Upload .dat")
    hea_file = st.sidebar.file_uploader("Upload .hea")
else:
    record_name = st.sidebar.text_input("MIT-BIH Record ID")

lowcut = st.sidebar.slider("Low Cutoff", 0.1, 5.0, 0.5)
highcut = st.sidebar.slider("High Cutoff", 20.0, 100.0, 40.0)
order = st.sidebar.slider("Filter Order", 2, 6, 3)

start = st.sidebar.button("Start Analysis")

# ================= SUBJECT INFO =================
st.markdown("## Subject Information")

col1, col2, col3 = st.columns(3)
name = col1.text_input("Name")
age = col2.number_input("Age", 0, 120)
gender = col3.selectbox("Gender", ["Male", "Female", "Other"])

st.markdown(f"""
<div class="card">
<b>Name:</b> {name}<br>
<b>Age:</b> {age}<br>
<b>Gender:</b> {gender}
</div>
""", unsafe_allow_html=True)

# ================= FILE INFO (ADDED) =================
def file_info(fs, raw):
    st.markdown(f"""
    <div class="card">
    <b>File Info</b><br>
    Sampling Rate: {fs} Hz<br>
    Samples: {len(raw)}<br>
    Duration: {len(raw)/fs:.2f} sec
    </div>
    """, unsafe_allow_html=True)

# ================= LOAD =================
def load_record():
    if zip_file:
        path = "tmp"
        os.makedirs(path, exist_ok=True)
        with zipfile.ZipFile(zip_file) as z:
            z.extractall(path)
        for f in os.listdir(path):
            if f.endswith(".hea"):
                return wfdb.rdrecord(os.path.join(path, f.replace(".hea","")))
    if dat_file and hea_file:
        open("temp.dat","wb").write(dat_file.read())
        open("temp.hea","wb").write(hea_file.read())
        return wfdb.rdrecord("temp")
    if record_name:
        return wfdb.rdrecord(record_name, pn_dir="mitdb")

# ================= FILTER =================
def bandpass(ecg, fs):
    ecg = ecg - np.mean(ecg)
    nyq = fs/2
    b,a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return filtfilt(b,a,ecg)

# ================= R PEAK =================
def detect_r(ecg, fs):
    sq = ecg**2
    win = int(0.12*fs)
    mwa = np.convolve(sq, np.ones(win)/win, mode='same')
    thresh = np.mean(mwa) + 0.5*np.std(mwa)
    peaks,_ = find_peaks(mwa, height=thresh, distance=int(0.25*fs))
    return peaks

def entropy(rr):
    h,_ = np.histogram(rr,bins=10,density=True)
    h += 1e-6
    return -np.sum(h*np.log(h))

# ================= RUN =================
if start:

    rec = load_record()
    fs = rec.fs
    raw = rec.p_signal[:,0]

    ecg = bandpass(raw, fs)
    rpeaks = detect_r(ecg, fs)
    rr = np.diff(rpeaks)/fs

    hr = 60/np.mean(rr)
    sdnn = np.std(rr)*1000
    rmssd = np.sqrt(np.mean(np.diff(rr)**2))*1000
    ent = entropy(rr)
    ectopic = np.sum((rr<0.3)|(rr>1.5))

    file_info(fs, raw)

    # ================= TABS =================
    tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8,tab9 = st.tabs([
        "Raw","Filtered","R Peaks","RR","Time","Frequency","Poincare","Stats","Summary"
    ])

    # ================= RAW =================
    with tab1:
        st.line_chart(raw)
        st.markdown(f"""
        <div class="card">
        Raw ECG shows full cardiac electrical activity with noise and artifacts.
        Useful for morphology but not direct clinical analysis.
        Sampling preserved at {fs} Hz with full signal integrity.
        </div>
        """, unsafe_allow_html=True)

    # ================= FILTERED =================
    with tab2:
        st.line_chart(ecg)
        st.markdown("""
        <div class="card">
        Bandpass filtered ECG enhances QRS complex and removes baseline wander.
        Signal is now suitable for R-peak detection and HRV extraction.
        Morphology preserved with improved clarity.
        </div>
        """, unsafe_allow_html=True)

    # ================= R PEAKS =================
    with tab3:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=ecg))
        fig.add_trace(go.Scatter(x=rpeaks,y=ecg[rpeaks],mode='markers'))
        st.plotly_chart(fig)

        st.markdown("""
        <div class="card">
        R-peaks represent ventricular depolarization.
        Accurate detection ensures correct RR interval computation.
        Each peak corresponds to one heartbeat cycle.
        </div>
        """, unsafe_allow_html=True)

    # ================= RR =================
    with tab4:
        st.line_chart(rr)
        st.markdown("""
        <div class="card">
        RR intervals reflect beat-to-beat variability controlled by autonomic nervous system.
        Irregular spacing indicates physiological modulation or ectopic activity.
        </div>
        """, unsafe_allow_html=True)

    # ================= TIME =================
    with tab5:
        st.plotly_chart(px.line(rr))
        st.dataframe(pd.DataFrame({
            "HR":[hr],
            "SDNN":[sdnn],
            "RMSSD":[rmssd]
        }))

    # ================= FREQUENCY (COLORED) =================
    with tab6:
        f,p = welch(rr-np.mean(rr))
        fig = go.Figure()
        fig.add_trace(go.Bar(x=f,y=p,name="PSD"))
        st.plotly_chart(fig)

    # ================= POINCARE =================
    with tab7:
        st.plotly_chart(go.Figure(data=go.Scatter(x=rr[:-1],y=rr[1:],mode='markers')))

    # ================= STATS =================
    with tab8:
        st.dataframe(pd.DataFrame({
            "Metric":["HR","SDNN","RMSSD","Entropy","Ectopic Beats"],
            "Value":[hr,sdnn,rmssd,ent,ectopic]
        }))
        st.plotly_chart(px.bar(x=["HR","SDNN","RMSSD","Entropy"],y=[hr,sdnn,rmssd,ent]))

    # ================= SUMMARY =================
    with tab9:
        st.markdown(f"""
        <div class="card">
        <h3>Clinical Summary</h3>
        Heart Rate: {hr:.2f}<br>
        HRV shows autonomic balance with variability index SDNN {sdnn:.2f}.<br>
        Entropy indicates signal complexity = {ent:.2f}.<br>
        Ectopic beats detected = {ectopic}.<br>
        Overall interpretation: {'Normal' if hr<100 else 'Stress/Arrhythmia risk'}.
        </div>
        """, unsafe_allow_html=True)

else:
    st.info("Upload ECG and start analysis")