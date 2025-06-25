import streamlit as st
import numpy as np
from PIL import Image, ImageSequence
import movement_detector
import io
import base64
import cv2
import tempfile

st.title("Gelişmiş Kamera & Nesne Hareketi Tespiti Demo")
st.write(
    "Bir video (mp4/gif) veya bir dizi görüntü yükleyin. Uygulama, kamera ve nesne hareketlerini daha sağlam şekilde tespit eder."
)

with st.sidebar:
    st.header("Ayarlar")
    cam_motion_thresh = st.slider(
        "Kamera Hareketi Eşiği (Optik Akış)", min_value=0.5, max_value=10.0, value=2.0, step=0.1
    )
    obj_area_thresh = st.slider(
        "Nesne Hareketi Alan Eşiği (px)", min_value=100, max_value=5000, value=500, step=50
    )
    show_cam_visuals = st.checkbox("Kamera Hareketi Görselleştirmelerini Göster", value=True)
    show_obj_visuals = st.checkbox("Nesne Hareketi Görselleştirmelerini Göster", value=True)
    max_show = st.number_input("En fazla kaç görsel gösterilsin?", min_value=1, max_value=50, value=10, step=1)

uploaded_files = st.file_uploader(
    "Video (mp4, gif) veya görüntü dosyalarını seçin",
    type=["jpg", "jpeg", "png", "mp4", "gif"],
    accept_multiple_files=True
)

def to_grayscale_if_needed(frame):
    if len(frame.shape) == 2:
        return np.stack([frame]*3, axis=-1)
    if frame.shape[-1] == 1:
        return np.concatenate([frame]*3, axis=-1)
    if frame.shape[-1] == 4:
        return frame[:, :, :3]
    return frame

@st.cache_data(show_spinner=False)
def extract_frames_from_video(file):
    frames = []
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name
    cap = cv2.VideoCapture(tmp_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = 300  # RAM patlamasını önlemek için sınır
    step = max(1, frame_count // max_frames) if frame_count > max_frames else 1
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        idx += 1
        if len(frames) >= max_frames:
            break
    cap.release()
    return frames

@st.cache_data(show_spinner=False)
def extract_frames_from_gif(file):
    frames = []
    image = Image.open(file)
    for i, frame in enumerate(ImageSequence.Iterator(image)):
        if i > 300:  # RAM patlamasını önle
            break
        arr = np.array(frame.convert("RGB"))
        frames.append(arr)
    return frames

@st.cache_data(show_spinner=False)
def extract_frames_from_images(files):
    frames = []
    for uploaded_file in files:
        image = Image.open(uploaded_file)
        frame = np.array(image)
        frame = to_grayscale_if_needed(frame)
        frames.append(frame)
    return frames

@st.cache_data(show_spinner=False)
def analyze_frames(all_frames, cam_motion_thresh, obj_area_thresh):
    cam_idx, obj_idx, cam_annotated, obj_annotated = [], [], [], []
    prev_gray = None
    fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=False)
    for idx, frame in enumerate(all_frames):
        if frame.shape[-1] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame[...,0].max()>1 else frame
        else:
            gray = frame if len(frame.shape)==2 else frame[...,0]
        # Kamera hareketi: optik akış
        cam_move = False
        cam_vis = frame.copy()
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            mean_mag = np.mean(mag)
            if mean_mag > cam_motion_thresh:
                cam_move = True
                step = 16
                for y in range(0, flow.shape[0], step):
                    for x in range(0, flow.shape[1], step):
                        fx, fy = flow[y, x]
                        cv2.arrowedLine(cam_vis, (x, y), (int(x+fx), int(y+fy)), (0,0,255), 1, tipLength=0.4)
        if cam_move:
            cam_idx.append(idx)
            cam_annotated.append(cam_vis)
        # Nesne hareketi: MOG2 + kontur
        fgmask = fgbg.apply(frame)
        th = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)[1]
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        obj_vis = frame.copy()
        found_obj = False
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > obj_area_thresh:
                found_obj = True
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(obj_vis, (x, y), (x+w, y+h), (0,255,0), 2)
                cv2.drawContours(obj_vis, [cnt], -1, (255,0,0), 1)
        if found_obj:
            obj_idx.append(idx)
            obj_annotated.append(obj_vis)
        prev_gray = gray
    return cam_idx, obj_idx, cam_annotated, obj_annotated

if uploaded_files:
    # Dosya tipine göre kareleri çıkar
    all_frames = []
    if all(f.name.lower().endswith((".jpg",".jpeg",".png")) for f in uploaded_files):
        all_frames = extract_frames_from_images(uploaded_files)
    else:
        for uploaded_file in uploaded_files:
            filename = uploaded_file.name.lower()
            if filename.endswith(".mp4"):
                frames = extract_frames_from_video(uploaded_file)
            elif filename.endswith(".gif"):
                frames = extract_frames_from_gif(uploaded_file)
            else:
                image = Image.open(uploaded_file)
                frame = np.array(image)
                frame = to_grayscale_if_needed(frame)
                frames = [frame]
            all_frames.extend(frames)

    st.write(f"{len(all_frames)} kare yüklendi.")
    with st.spinner("Analiz yapılıyor, lütfen bekleyin..."):
        cam_idx, obj_idx, cam_annotated, obj_annotated = analyze_frames(all_frames, cam_motion_thresh, obj_area_thresh)
    st.write(f"Kamera hareketi tespit edilen kareler: {cam_idx}")
    st.write(f"Nesne hareketi tespit edilen kareler: {obj_idx}")

    def get_download_link(indices, label):
        indices_str = ",".join(map(str, indices))
        b = io.BytesIO()
        b.write(indices_str.encode())
        b.seek(0)
        b64 = base64.b64encode(b.read()).decode()
        href = f'<a href="data:file/txt;base64,{b64}" download="{label}_indices.txt">{label} İndekslerini İndir</a>'
        return href

    st.markdown(get_download_link(cam_idx, "kamera_hareket"), unsafe_allow_html=True)
    st.markdown(get_download_link(obj_idx, "nesne_hareket"), unsafe_allow_html=True)

    if show_cam_visuals and cam_annotated:
        st.subheader("Kamera Hareketi Görselleştirmeleri (Optik Akış)")
        for idx, annotated in list(zip(cam_idx, cam_annotated))[:max_show]:
            st.image(annotated, caption=f"Kamera hareketi: Kare {idx}", use_container_width=True)
    if show_obj_visuals and obj_annotated:
        st.subheader("Nesne Hareketi Görselleştirmeleri (Kontur & Kutu)")
        for idx, annotated in list(zip(obj_idx, obj_annotated))[:max_show]:
            st.image(annotated, caption=f"Nesne hareketi: Kare {idx}", use_container_width=True)
