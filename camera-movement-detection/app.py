import streamlit as st
import numpy as np
from PIL import Image, ImageSequence
import movement_detector
import io
import base64
import cv2
import tempfile
import gc

st.title("Gelişmiş Kamera & Nesne Hareketi Tespiti Demo (Optimize)")
st.write(
    "Bir video (mp4/gif) veya bir dizi görüntü yükleyin. Uygulama, kamera ve nesne hareketlerini daha sağlam şekilde tespit eder. RAM dostu!"
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

def process_video_file(file, cam_motion_thresh, obj_area_thresh, show_cam_visuals, show_obj_visuals):
    cam_idx, obj_idx = [], []
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name
    cap = cv2.VideoCapture(tmp_path)
    prev_gray = None
    fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=False)
    idx = 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress = st.empty()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        progress.info(f"Analiz edilen kare: {idx+1}/{total}")
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
                if show_cam_visuals:
                    step = 16
                    for y in range(0, flow.shape[0], step):
                        for x in range(0, flow.shape[1], step):
                            fx, fy = flow[y, x]
                            cv2.arrowedLine(cam_vis, (x, y), (int(x+fx), int(y+fy)), (0,0,255), 1, tipLength=0.4)
        if cam_move:
            cam_idx.append(idx)
            if show_cam_visuals:
                st.image(cam_vis, caption=f"Kamera hareketi: Kare {idx}", use_container_width=True)
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
            if show_obj_visuals:
                st.image(obj_vis, caption=f"Nesne hareketi: Kare {idx}", use_container_width=True)
        prev_gray = gray
        idx += 1
        del frame, cam_vis, obj_vis, fgmask, th, contours
        gc.collect()
    progress.empty()
    cap.release()
    return cam_idx, obj_idx

def process_gif_file(file, cam_motion_thresh, obj_area_thresh, show_cam_visuals, show_obj_visuals):
    cam_idx, obj_idx = [], []
    image = Image.open(file)
    frames = ImageSequence.Iterator(image)
    prev_gray = None
    fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=False)
    total = image.n_frames if hasattr(image, 'n_frames') else 0
    progress = st.empty()
    for idx, frame in enumerate(frames):
        arr = np.array(frame.convert("RGB"))
        progress.info(f"Analiz edilen kare: {idx+1}/{total}")
        if arr.shape[-1] == 3:
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if arr[...,0].max()>1 else arr
        else:
            gray = arr if len(arr.shape)==2 else arr[...,0]
        cam_move = False
        cam_vis = arr.copy()
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            mean_mag = np.mean(mag)
            if mean_mag > cam_motion_thresh:
                cam_move = True
                if show_cam_visuals:
                    step = 16
                    for y in range(0, flow.shape[0], step):
                        for x in range(0, flow.shape[1], step):
                            fx, fy = flow[y, x]
                            cv2.arrowedLine(cam_vis, (x, y), (int(x+fx), int(y+fy)), (0,0,255), 1, tipLength=0.4)
        if cam_move:
            cam_idx.append(idx)
            if show_cam_visuals:
                st.image(cam_vis, caption=f"Kamera hareketi: Kare {idx}", use_container_width=True)
        fgmask = fgbg.apply(arr)
        th = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)[1]
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        obj_vis = arr.copy()
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
            if show_obj_visuals:
                st.image(obj_vis, caption=f"Nesne hareketi: Kare {idx}", use_container_width=True)
        prev_gray = gray
        del arr, cam_vis, obj_vis, fgmask, th, contours
        gc.collect()
    progress.empty()
    return cam_idx, obj_idx

def process_image_file(uploaded_file, cam_motion_thresh, obj_area_thresh, show_cam_visuals, show_obj_visuals):
    cam_idx, obj_idx = [], []
    image = Image.open(uploaded_file)
    frame = np.array(image)
    frame = to_grayscale_if_needed(frame)
    idx = 0
    prev_gray = None
    fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=False)
    if frame.shape[-1] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame[...,0].max()>1 else frame
    else:
        gray = frame if len(frame.shape)==2 else frame[...,0]
    cam_move = False
    cam_vis = frame.copy()
    # Tek karede kamera hareketi anlamlı değil, sadece nesne hareketi bakılabilir
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
        if show_obj_visuals:
            st.image(obj_vis, caption=f"Nesne hareketi: Kare {idx}", use_container_width=True)
    del frame, cam_vis, obj_vis, fgmask, th, contours
    gc.collect()
    return cam_idx, obj_idx

if uploaded_files:
    all_cam_idx, all_obj_idx = [], []
    for uploaded_file in uploaded_files:
        filename = uploaded_file.name.lower()
        if filename.endswith(".mp4"):
            cam_idx, obj_idx = process_video_file(uploaded_file, cam_motion_thresh, obj_area_thresh, show_cam_visuals, show_obj_visuals)
        elif filename.endswith(".gif"):
            cam_idx, obj_idx = process_gif_file(uploaded_file, cam_motion_thresh, obj_area_thresh, show_cam_visuals, show_obj_visuals)
        else:
            cam_idx, obj_idx = process_image_file(uploaded_file, cam_motion_thresh, obj_area_thresh, show_cam_visuals, show_obj_visuals)
        all_cam_idx.extend(cam_idx)
        all_obj_idx.extend(obj_idx)
    st.write(f"Kamera hareketi tespit edilen kareler: {all_cam_idx}")
    st.write(f"Nesne hareketi tespit edilen kareler: {all_obj_idx}")

    def get_download_link(indices, label):
        indices_str = ",".join(map(str, indices))
        b = io.BytesIO()
        b.write(indices_str.encode())
        b.seek(0)
        b64 = base64.b64encode(b.read()).decode()
        href = f'<a href="data:file/txt;base64,{b64}" download="{label}_indices.txt">{label} İndekslerini İndir</a>'
        return href

    st.markdown(get_download_link(all_cam_idx, "kamera_hareket"), unsafe_allow_html=True)
    st.markdown(get_download_link(all_obj_idx, "nesne_hareket"), unsafe_allow_html=True)