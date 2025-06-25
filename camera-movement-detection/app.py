import streamlit as st
import numpy as np
from PIL import Image, ImageSequence
import movement_detector as md              # tek satırda içe aktar
import cv2
import tempfile, io, base64, gc

st.set_page_config(page_title="Hareket Tespiti Demo", layout="wide")
st.title("Gelişmiş Kamera & Nesne Hareketi Tespiti Demo")
st.write("Bir video (mp4/gif) veya görüntü dizisi yükleyin. Uygulama kamera ve nesne hareketlerini tespit eder.")

# ---------- Yan Panel ---------- #
with st.sidebar:
    st.header("Ayarlar")
    downscale = st.slider("Çözünürlük Ölçeği", 0.25, 1.0, 0.5, 0.05)
    frame_step = st.slider("İşlenecek her N. kare", 1, 10, 2, 1)
    cam_motion_thresh = st.slider("Kamera Hareketi Eşiği (Optik Akış)", 0.5, 10.0, 2.0, 0.1)
    obj_area_thresh = st.slider("Nesne Hareketi Alan Eşiği (px)", 100, 5000, 500, 50)
    show_cam_visuals = st.checkbox("Kamera Hareketi Görselleştirmelerini Göster", True)
    show_obj_visuals = st.checkbox("Nesne Hareketi Görselleştirmelerini Göster", True)

# ---------- Dosya Yükleyici ---------- #
uploaded = st.file_uploader(
    "Video (mp4, gif) veya görüntü dosyaları (.png/.jpg) seçin",
    type=["jpg","jpeg","png","mp4","gif"], accept_multiple_files=True
)

# ---------- Önbellekli karem çıkarıcılar ---------- #
@st.cache_data(show_spinner=False)
def video_frames(file_bytes: bytes, scale: float, step: int):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(file_bytes); path = tmp.name
    cap, out = cv2.VideoCapture(path), []
    idx = 0
    while True:
        ret, frm = cap.read()
        if not ret:
            break
        if idx % step == 0:
            if scale != 1.0:
                w = int(frm.shape[1]*scale); h = int(frm.shape[0]*scale)
                frm = cv2.resize(frm, (w, h))
            out.append(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
        idx += 1
    cap.release()
    return out

@st.cache_data(show_spinner=False)
def gif_frames(file, scale: float, step: int):
    img = Image.open(file); out = []
    for i, frame in enumerate(ImageSequence.Iterator(img)):
        if i % step == 0:
            arr = np.array(frame.convert("RGB"))
            if scale != 1.0:
                h,w = arr.shape[:2]
                arr = cv2.resize(arr, (int(w*scale), int(h*scale)))
            out.append(arr)
    return out

# ---------- Yardımcı ---------- #
def get_download_link(indices, label):
    txt = ",".join(map(str, indices)).encode()
    b64 = base64.b64encode(txt).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{label}_indices.txt">{label} İndekslerini İndir</a>'

# ---------- Ana Akış ---------- #
if uploaded:
    all_frames = []
    for up in uploaded:
        name = up.name.lower()
        if name.endswith(".mp4"):
            all_frames += video_frames(up.getbuffer(), downscale, frame_step)
        elif name.endswith(".gif"):
            all_frames += gif_frames(up, downscale, frame_step)
        else:
            img = np.array(Image.open(up).convert("RGB"))
            if downscale != 1.0:
                h,w = img.shape[:2]
                img = cv2.resize(img, (int(w*downscale), int(h*downscale)))
            all_frames.append(img)

    total = len(all_frames)
    if total == 0:
        st.error("Hiç kare bulunamadı!")
        st.stop()

    st.success(f"{total} kare yüklendi (ölçek = {downscale}, adım = {frame_step}).")
    prog = st.progress(0, text="Analiz başlatılıyor…")

    cam_idx, obj_idx, cam_vis, obj_vis = [], [], [], []
    prev_gray = None
    fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=False)

    for i, frm in enumerate(all_frames):
        gray = md.preprocess_frame(frm)                # güvenli çağrı
        # ---- Kamera Hareketi (Optik Akış) ---- #
        if prev_gray is not None:
            flow  = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, .5,3,15,3,5,1.2,0)
            mean  = np.mean(cv2.cartToPolar(flow[...,0], flow[...,1])[0])
            if mean > cam_motion_thresh:
                cam_idx.append(i)
                if show_cam_visuals:
                    vis = frm.copy()
                    step = 16
                    for y in range(0, flow.shape[0], step):
                        for x in range(0, flow.shape[1], step):
                            fx, fy = flow[y,x]
                            cv2.arrowedLine(vis, (x,y), (int(x+fx),int(y+fy)), (0,0,255), 1, tipLength=.4)
                    cam_vis.append(vis)
        prev_gray = gray

        # ---- Nesne Hareketi (MOG2) ---- #
        fg   = fgbg.apply(frm)
        mask = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)[1]
        cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        found = False
        if cnts:
            vis = frm.copy()
            for c in cnts:
                area = cv2.contourArea(c)
                if area > obj_area_thresh:
                    found = True
                    x,y,w,h = cv2.boundingRect(c)
                    cv2.rectangle(vis,(x,y),(x+w,y+h),(0,255,0),2)
            if found:
                obj_idx.append(i)
                if show_obj_visuals:
                    obj_vis.append(vis)

        # ---- Progress ---- #
        if (i+1) % 5 == 0 or i+1 == total:
            prog.progress((i+1)/total, text=f"Kare {i+1}/{total}")

    prog.empty()
    gc.collect()

    # ---------- Sonuç ---------- #
    st.subheader("Tespitler")
    st.write(f"Kamera hareketi kareleri: {cam_idx}")
    st.markdown(get_download_link(cam_idx, "kamera_hareket"), unsafe_allow_html=True)
    st.write(f"Nesne hareketi kareleri: {obj_idx}")
    st.markdown(get_download_link(obj_idx, "nesne_hareket"), unsafe_allow_html=True)

    if show_cam_visuals and cam_vis:
        st.subheader("Kamera Hareketi Görselleri")
        for i, v in zip(cam_idx, cam_vis):
            st.image(v, caption=f"Kare {i}", use_container_width=True)

    if show_obj_visuals and obj_vis:
        st.subheader("Nesne Hareketi Görselleri")
        for i, v in zip(obj_idx, obj_vis):
            st.image(v, caption=f"Kare {i}", use_container_width=True)
