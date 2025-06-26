import cv2 
import numpy as np
from typing import List, Optional, Tuple


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    Verilen bir görüntü karesini ön işler (ör. gri tonlamaya çevirir).
    Args:
        frame: BGR formatında bir numpy görüntü karesi.
    Returns:
        Gri tonlamalı numpy görüntü karesi.
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def frame_difference(prev_frame: np.ndarray, curr_frame: np.ndarray) -> np.ndarray:
    """
    İki ardışık kare arasındaki mutlak farkı hesaplar.
    Args:
        prev_frame: Önceki gri tonlamalı kare.
        curr_frame: Şimdiki gri tonlamalı kare.
    Returns:
        İki kare arasındaki mutlak fark görüntüsü.
    """
    return cv2.absdiff(prev_frame, curr_frame)


def calculate_difference_score(diff: np.ndarray) -> float:
    """
    Fark görüntüsünün ortalama skorunu hesaplar.
    Args:
        diff: İki kare arasındaki mutlak fark görüntüsü.
    Returns:
        Fark skorunun ortalaması (float).
    """
    return float(np.mean(diff))


def detect_significant_movement(frames: List[np.ndarray], threshold: float = 50.0) -> List[int]:
    """
    Kamerada anlamlı hareket olan karelerin indekslerini tespit eder.
    Args:
        frames: Görüntü karelerinin listesi (numpy array olarak).
        threshold: Hareket tespiti için hassasiyet eşiği.
    Returns:
        Anlamlı hareket tespit edilen karelerin indekslerinin listesi.
    """
    movement_indices: List[int] = []
    prev_gray: Optional[np.ndarray] = None
    for idx, frame in enumerate(frames):
        gray = preprocess_frame(frame)
        if prev_gray is not None:
            diff = frame_difference(prev_gray, gray)
            score = calculate_difference_score(diff)
            if score > threshold:
                movement_indices.append(idx)
        prev_gray = gray
    return movement_indices


def detect_significant_movement_orb(
    frames: List[np.ndarray], threshold: float = 10.0, min_matches: int = 10
) -> List[int]:
    """
    ORB ve homografi kullanarak global kamera hareketini tespit eder.
    Args:
        frames: Görüntü karelerinin listesi (numpy array olarak).
        threshold: Hareket tespiti için homografi dönüşüm büyüklüğü eşiği.
        min_matches: Homografi için gerekli minimum eşleşme sayısı.
    Returns:
        Anlamlı hareket tespit edilen karelerin indekslerinin listesi.
    """
    movement_indices: List[int] = []
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    prev_kp, prev_des = None, None
    for idx, frame in enumerate(frames):
        gray = preprocess_frame(frame)
        kp, des = orb.detectAndCompute(gray, None)
        if prev_kp is not None and prev_des is not None and des is not None:
            matches = bf.match(prev_des, des)
            matches = sorted(matches, key=lambda x: x.distance)
            if len(matches) >= min_matches:
                src_pts = np.float32([prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                try:
                    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    if H is not None:
                        dx = H[0, 2]
                        dy = H[1, 2]
                        da = np.arctan2(H[1, 0], H[0, 0])
                        motion_magnitude = np.sqrt(dx ** 2 + dy ** 2) + abs(da)
                        if motion_magnitude > threshold:
                            movement_indices.append(idx)
                except cv2.error:
                    pass  # Homografi bulunamazsa atla
        prev_kp, prev_des = kp, des
    return movement_indices


def annotate_camera_movement(
    frame1: np.ndarray,
    frame2: np.ndarray,
    kp1: List[cv2.KeyPoint],
    kp2: List[cv2.KeyPoint],
    matches: List[cv2.DMatch],
    H: np.ndarray = None,
    max_points: int = 30
) -> np.ndarray:
    """
    İki kare arasındaki eşleşen anahtar noktaları ve (varsa) homografi vektörlerini çizer.
    Args:
        frame1: Önceki kare (BGR).
        frame2: Şimdiki kare (BGR).
        kp1: Önceki karenin anahtar noktaları.
        kp2: Şimdiki karenin anahtar noktaları.
        matches: Eşleşen noktalar listesi.
        H: Homografi matrisi (isteğe bağlı).
        max_points: Maksimum çizilecek eşleşme sayısı.
    Returns:
        frame2 üzerine çizilmiş görselleştirme (BGR).
    """
    annotated = frame2.copy()
    draw_matches = matches[:max_points]
    for m in draw_matches:
        pt1 = tuple(map(int, kp1[m.queryIdx].pt))
        pt2 = tuple(map(int, kp2[m.trainIdx].pt))
        cv2.circle(annotated, pt2, 4, (0, 255, 0), -1)
        cv2.line(annotated, pt1, pt2, (255, 0, 0), 1)
    # Homografi varsa, frame1 köşelerini frame2'ye projekte et
    if H is not None:
        h, w = frame1.shape[:2]
        corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
        projected = cv2.perspectiveTransform(corners, H)
        projected = np.int32(projected)
        cv2.polylines(annotated, [projected], True, (0,0,255), 2)
    return annotated


def detect_camera_and_object_movement(
    frames: List[np.ndarray],
    homography_threshold: float = 10.0,
    diff_threshold: float = 50.0,
    min_matches: int = 10
) -> Tuple[List[int], List[int], List[np.ndarray]]:
    """
    Hem kamera (global) hem de nesne (lokal) hareketini tespit eder ve kamera hareketi olan karelerde görsel açıklama döner.
    Args:
        frames: Görüntü karelerinin listesi (numpy array olarak).
        homography_threshold: Kamera hareketi için homografi büyüklüğü eşiği.
        diff_threshold: Nesne hareketi için kare farkı eşiği.
        min_matches: Homografi için gerekli minimum eşleşme sayısı.
    Returns:
        (kamera_hareket_indeksleri, nesne_hareket_indeksleri, kamera_hareket_annotated_frames)
    """
    camera_movement_indices: List[int] = []
    object_movement_indices: List[int] = []
    camera_annotated_frames: List[np.ndarray] = []
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    prev_kp, prev_des = None, None
    prev_gray = None
    prev_frame = None
    for idx, frame in enumerate(frames):
        gray = preprocess_frame(frame)
        kp, des = orb.detectAndCompute(gray, None)
        local_change = False
        global_change = False
        diff_score = None
        motion_magnitude = None
        matches = []
        H = None
        if prev_gray is not None:
            diff = frame_difference(prev_gray, gray)
            diff_score = calculate_difference_score(diff)
            if diff_score > diff_threshold:
                local_change = True
        if prev_kp is not None and prev_des is not None and des is not None:
            matches = bf.match(prev_des, des)
            matches = sorted(matches, key=lambda x: x.distance)
            if len(matches) >= min_matches:
                src_pts = np.float32([prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                try:
                    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    if H is not None:
                        dx = H[0, 2]
                        dy = H[1, 2]
                        da = np.arctan2(H[1, 0], H[0, 0])
                        motion_magnitude = np.sqrt(dx ** 2 + dy ** 2) + abs(da)
                        if motion_magnitude > homography_threshold:
                            global_change = True
                except cv2.error:
                    pass
        # Sınıflandırma
        if global_change and local_change:
            camera_movement_indices.append(idx)
            if prev_frame is not None:
                annotated = annotate_camera_movement(prev_frame, frame, prev_kp, kp, matches, H)
                camera_annotated_frames.append(annotated)
        elif local_change and not global_change:
            object_movement_indices.append(idx)
        prev_kp, prev_des = kp, des
        prev_gray = gray
        prev_frame = frame
    return camera_movement_indices, object_movement_indices, camera_annotated_frames


def detect_motion_robust(
    frames: List[np.ndarray],
    cam_motion_thresh: float = 2.0,
    obj_area_thresh: int = 500
) -> Tuple[List[int], List[int], List[np.ndarray], List[np.ndarray]]:
    """
    Optik akış ile kamera hareketi, MOG2+kontur ile nesne hareketi tespit eder.
    Args:
        frames: Kare listesi (RGB veya BGR).
        cam_motion_thresh: Kamera hareketi için ortalama optik akış büyüklüğü eşiği.
        obj_area_thresh: Nesne hareketi için minimum kontur alanı.
    Returns:
        (kamera_hareket_indeksleri, nesne_hareket_indeksleri, kamera_annotated, nesne_annotated)
    """
    camera_movement_indices = []
    object_movement_indices = []
    camera_annotated = []
    object_annotated = []
    prev_gray = None
    fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=False)
    for idx, frame in enumerate(frames):
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
                # Görselleştirme: optik akış vektörleri
                step = 16
                for y in range(0, flow.shape[0], step):
                    for x in range(0, flow.shape[1], step):
                        fx, fy = flow[y, x]
                        cv2.arrowedLine(cam_vis, (x, y), (int(x+fx), int(y+fy)), (0,0,255), 1, tipLength=0.4)
        if cam_move:
            camera_movement_indices.append(idx)
            camera_annotated.append(cam_vis)
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
            object_movement_indices.append(idx)
            object_annotated.append(obj_vis)
        prev_gray = gray
    return camera_movement_indices, object_movement_indices, camera_annotated, object_annotated
