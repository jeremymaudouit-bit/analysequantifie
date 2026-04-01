
import io
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image
from fpdf import FPDF

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="Biomeca Suite - Analyse unifiée", layout="wide")
st.title("🧍🏃 Biomeca Suite - Analyse unifiée")
st.caption("Une seule application pour l'analyse cinématique, frontale et posturale avec un rapport global unique.")

FPS_DEFAULT = 30
POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS

# ============================================================
# MEDIAPIPE
# ============================================================
@st.cache_resource
def load_video_pose():
    return mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

@st.cache_resource
def load_image_pose():
    return mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

POSE_VIDEO = load_video_pose()
POSE_IMAGE = load_image_pose()

# ============================================================
# PARAMÈTRES COMMUNS
# ============================================================
with st.sidebar:
    st.header("Paramètres communs")
    nom = st.text_input("Nom", "")
    prenom = st.text_input("Prénom", "")
    taille_cm = st.number_input("Taille du patient (cm)", min_value=80, max_value=230, value=170, step=1)
    conf = st.slider("Seuil de visibilité minimal", 0.1, 0.95, 0.5, 0.05)
    smooth = st.slider("Lissage", 1, 15, 5, 2)
    show_norm = st.checkbox("Afficher les courbes indicatives de référence", value=True)
    camera_pos = st.selectbox("Côté / angle vidéo de profil", ["Profil droit", "Profil gauche"])
    phase_cote = st.selectbox("Phases / côté dominant", ["Aucune", "Droite", "Gauche", "Les deux"])
    sample_stride = st.slider("Pas d'échantillonnage vidéo (1 = toutes les images)", 1, 6, 2)
    num_photos = st.slider("Nombre d'images annotées à exporter", 1, 8, 3)
    max_frames = st.slider("Nombre maximal d'images analysées par vidéo", 60, 600, 180, 30)

patient = {
    "nom": nom.strip(),
    "prenom": prenom.strip(),
    "taille_cm": int(taille_cm),
    "conf": float(conf),
    "smooth": int(smooth),
    "show_norm": bool(show_norm),
    "camera_pos": camera_pos,
    "phase_cote": phase_cote,
    "sample_stride": int(sample_stride),
    "num_photos": int(num_photos),
    "max_frames": int(max_frames),
}

# ============================================================
# UPLOADS
# ============================================================
st.subheader("1) Chargement des fichiers")
c1, c2 = st.columns(2)
with c1:
    file_video_cinematique = st.file_uploader("Vidéo profil - analyse cinématique", type=["mp4", "mov", "avi"], key="cin")
    file_video_frontale = st.file_uploader("Vidéo face / arrière - analyse frontale", type=["mp4", "mov", "avi"], key="front")
with c2:
    file_image_frontale = st.file_uploader("Photo posturale frontale", type=["png", "jpg", "jpeg"], key="img_front")
    file_image_laterale = st.file_uploader("Photo posturale latérale", type=["png", "jpg", "jpeg"], key="img_lat")

# ============================================================
# OUTILS GÉNÉRAUX
# ============================================================
def save_uploaded_file(uploaded_file) -> Optional[str]:
    if uploaded_file is None:
        return None
    suffix = os.path.splitext(uploaded_file.name)[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.close()
    return tmp.name

def cleanup_tmp(paths: List[Optional[str]]) -> None:
    for path in paths:
        if path and os.path.exists(path):
            try:
                os.unlink(path)
            except OSError:
                pass

def ensure_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def rotate_if_landscape(img_rgb: np.ndarray) -> np.ndarray:
    if img_rgb.shape[1] > img_rgb.shape[0]:
        img_rgb = cv2.rotate(img_rgb, cv2.ROTATE_90_CLOCKWISE)
    return img_rgb

def safe_name(text: str) -> str:
    txt = (text or "").strip().replace(" ", "_")
    return txt or "patient"

def pdf_safe(text: Any) -> str:
    if text is None:
        return ""
    s = str(text)
    replacements = {
        "–": "-",
        "—": "-",
        "’": "'",
        "“": '"',
        "”": '"',
        "•": "-",
        "✅": "[OK]",
        "⚠️": "[!]",
        "é": "e",
        "è": "e",
        "ê": "e",
        "ë": "e",
        "à": "a",
        "â": "a",
        "î": "i",
        "ï": "i",
        "ô": "o",
        "ö": "o",
        "ù": "u",
        "û": "u",
        "ç": "c",
        "°": " deg",
    }
    for a, b in replacements.items():
        s = s.replace(a, b)
    return s

def interp_nan(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    idx = np.arange(len(arr))
    ok = ~np.isnan(arr)
    if ok.sum() >= 2:
        return np.interp(idx, idx[ok], arr[ok])
    return np.zeros_like(arr)

def smooth_ma(y: np.ndarray, win: int = 5) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if len(y) == 0:
        return y
    if win <= 1:
        return y
    if win % 2 == 0:
        win += 1
    pad = win // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(win) / win
    return np.convolve(ypad, kernel, mode="valid")

def nanmean(x: List[float]) -> float:
    arr = np.asarray(x, dtype=float)
    return float(np.nanmean(arr)) if len(arr) else float("nan")

def nanstd(x: List[float]) -> float:
    arr = np.asarray(x, dtype=float)
    return float(np.nanstd(arr)) if len(arr) else float("nan")

def angle_3pts(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)
    ba = a - b
    bc = c - b
    den = np.linalg.norm(ba) * np.linalg.norm(bc)
    if den < 1e-9:
        return np.nan
    cosang = np.clip(np.dot(ba, bc) / den, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

def orientation_angle(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    v = np.array([p2[0] - p1[0], p2[1] - p1[1]], dtype=float)
    if np.linalg.norm(v) < 1e-9:
        return np.nan
    return float(np.degrees(np.arctan2(v[1], v[0])))

def tilt_from_horizontal(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    ang = orientation_angle(p1, p2)
    if np.isnan(ang):
        return np.nan
    return float(ang)

def tilt_from_vertical(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    ang = orientation_angle(p1, p2)
    if np.isnan(ang):
        return np.nan
    return float(90 - ang)

def xy(landmark, width: int, height: int) -> Tuple[float, float]:
    return landmark.x * width, landmark.y * height

def vis(landmark) -> float:
    return float(getattr(landmark, "visibility", 1.0))

def avg_point(points: List[Tuple[float, float]]) -> Tuple[float, float]:
    arr = np.asarray(points, dtype=float)
    return float(arr[:, 0].mean()), float(arr[:, 1].mean())

def draw_landmarks_basic(image_bgr: np.ndarray, landmarks, width: int, height: int, conf: float) -> np.ndarray:
    out = image_bgr.copy()
    if landmarks is None:
        return out
    for a, b in POSE_CONNECTIONS:
        la = landmarks.landmark[a]
        lb = landmarks.landmark[b]
        if vis(la) >= conf and vis(lb) >= conf:
            pa = tuple(int(v) for v in xy(la, width, height))
            pb = tuple(int(v) for v in xy(lb, width, height))
            cv2.line(out, pa, pb, (0, 255, 0), 2)
    for lm in landmarks.landmark:
        if vis(lm) >= conf:
            p = tuple(int(v) for v in xy(lm, width, height))
            cv2.circle(out, p, 4, (0, 0, 255), -1)
    return out

def build_plot(series_dict: Dict[str, np.ndarray], title: str, y_label: str, show_norm: bool = False) -> str:
    fig = plt.figure(figsize=(9, 4))
    for label, values in series_dict.items():
        if values is None or len(values) == 0:
            continue
        plt.plot(values, label=label)
    plt.title(title)
    plt.xlabel("Images analysées")
    plt.ylabel(y_label)
    plt.grid(True, alpha=0.25)
    plt.legend()
    path = os.path.join(tempfile.gettempdir(), f"{title.replace(' ', '_').replace('/', '_')}.png")
    plt.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path

def read_video_frames(video_path: str, stride: int = 2, max_frames: int = 180) -> Tuple[List[np.ndarray], float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], FPS_DEFAULT
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 1 else FPS_DEFAULT
    frames = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % max(1, stride) == 0:
            frames.append(frame)
            if len(frames) >= max_frames:
                break
        idx += 1
    cap.release()
    return frames, float(fps)

def analyze_pose_frame(frame_bgr: np.ndarray, static: bool = False):
    rgb = ensure_rgb(frame_bgr)
    pose_obj = POSE_IMAGE if static else POSE_VIDEO
    return pose_obj.process(rgb)

def side_indices(camera_pos: str) -> Dict[str, int]:
    mp_pl = mp.solutions.pose.PoseLandmark
    if camera_pos == "Profil gauche":
        side = {
            "shoulder": mp_pl.LEFT_SHOULDER.value,
            "hip": mp_pl.LEFT_HIP.value,
            "knee": mp_pl.LEFT_KNEE.value,
            "ankle": mp_pl.LEFT_ANKLE.value,
            "heel": mp_pl.LEFT_HEEL.value,
            "foot": mp_pl.LEFT_FOOT_INDEX.value,
            "ear": mp_pl.LEFT_EAR.value,
        }
    else:
        side = {
            "shoulder": mp_pl.RIGHT_SHOULDER.value,
            "hip": mp_pl.RIGHT_HIP.value,
            "knee": mp_pl.RIGHT_KNEE.value,
            "ankle": mp_pl.RIGHT_ANKLE.value,
            "heel": mp_pl.RIGHT_HEEL.value,
            "foot": mp_pl.RIGHT_FOOT_INDEX.value,
            "ear": mp_pl.RIGHT_EAR.value,
        }
    return side

def default_result(title: str) -> Dict[str, Any]:
    return {
        "title": title,
        "status": "non_analyse",
        "summary": "Non analysé",
        "bullet_points": ["Aucun fichier n'a été fourni pour ce module."],
        "plots": [],
        "annotated_images": [],
        "metrics": {},
    }

# ============================================================
# ANALYSE CINÉMATIQUE
# ============================================================
def run_cinematic(video_path: Optional[str], patient: Dict[str, Any]) -> Dict[str, Any]:
    if not video_path:
        return default_result("Analyse cinématique")

    frames, fps = read_video_frames(video_path, patient["sample_stride"], patient["max_frames"])
    if not frames:
        return {
            "title": "Analyse cinématique",
            "status": "erreur",
            "summary": "Impossible de lire la vidéo.",
            "bullet_points": ["Le fichier vidéo n'a pas pu être décodé."],
            "plots": [],
            "annotated_images": [],
            "metrics": {},
        }

    idxs = side_indices(patient["camera_pos"])
    hip_angles, knee_angles, ankle_angles, trunk_tilts, heel_y = [], [], [], [], []
    annotations = []

    for frame in frames:
        result = analyze_pose_frame(frame, static=False)
        if not result.pose_landmarks:
            hip_angles.append(np.nan)
            knee_angles.append(np.nan)
            ankle_angles.append(np.nan)
            trunk_tilts.append(np.nan)
            heel_y.append(np.nan)
            continue

        lms = result.pose_landmarks.landmark
        h, w = frame.shape[:2]

        pts = {}
        all_ok = True
        for key, ind in idxs.items():
            lm = lms[ind]
            if vis(lm) < patient["conf"]:
                all_ok = False
            pts[key] = xy(lm, w, h)

        if all_ok:
            hip_angles.append(angle_3pts(pts["shoulder"], pts["hip"], pts["knee"]))
            knee_angles.append(angle_3pts(pts["hip"], pts["knee"], pts["ankle"]))
            ankle_angles.append(angle_3pts(pts["knee"], pts["ankle"], pts["foot"]))
            trunk_tilts.append(tilt_from_vertical(pts["hip"], pts["shoulder"]))
            heel_y.append(pts["heel"][1])
        else:
            hip_angles.append(np.nan)
            knee_angles.append(np.nan)
            ankle_angles.append(np.nan)
            trunk_tilts.append(np.nan)
            heel_y.append(np.nan)

    hip_s = smooth_ma(interp_nan(np.array(hip_angles)), patient["smooth"])
    knee_s = smooth_ma(interp_nan(np.array(knee_angles)), patient["smooth"])
    ankle_s = smooth_ma(interp_nan(np.array(ankle_angles)), patient["smooth"])
    trunk_s = smooth_ma(interp_nan(np.array(trunk_tilts)), patient["smooth"])
    heel_s = smooth_ma(interp_nan(np.array(heel_y)), patient["smooth"])

    for idx in np.linspace(0, len(frames) - 1, min(patient["num_photos"], len(frames)), dtype=int):
        r = analyze_pose_frame(frames[idx], static=False)
        img = draw_landmarks_basic(frames[idx], r.pose_landmarks, frames[idx].shape[1], frames[idx].shape[0], patient["conf"])
        out = os.path.join(tempfile.gettempdir(), f"cin_annot_{idx}.png")
        cv2.imwrite(out, img)
        annotations.append(out)

    contacts = []
    if len(heel_s) >= 3:
        for i in range(1, len(heel_s) - 1):
            if heel_s[i] > heel_s[i - 1] and heel_s[i] >= heel_s[i + 1]:
                contacts.append(i)

    step_times = np.diff(np.array(contacts) / max(fps, 1))
    plot1 = build_plot(
        {"Hanche": hip_s, "Genou": knee_s, "Cheville": ankle_s, "Tronc": trunk_s},
        "Cinematique_angles",
        "Angle (deg)",
        patient["show_norm"],
    )
    plot2 = build_plot({"Talon": heel_s}, "Cinematique_contact_talon", "Position Y (px)", False)

    bullets = [
        f"Images analysées : {len(frames)}",
        f"FPS estimé : {fps:.1f}",
        f"Amplitude hanche : {np.nanmax(hip_s) - np.nanmin(hip_s):.1f} deg",
        f"Amplitude genou : {np.nanmax(knee_s) - np.nanmin(knee_s):.1f} deg",
        f"Amplitude cheville : {np.nanmax(ankle_s) - np.nanmin(ankle_s):.1f} deg",
        f"Inclinaison moyenne du tronc : {np.nanmean(trunk_s):.1f} deg",
    ]
    if len(step_times):
        bullets.append(f"Temps moyen entre contacts détectés : {np.mean(step_times):.2f} s")
    else:
        bullets.append("Temps de pas non estimable sur cette vidéo.")

    return {
        "title": "Analyse cinématique",
        "status": "ok",
        "summary": "Analyse du profil réalisée à partir des angles de hanche, genou, cheville et de l'inclinaison du tronc.",
        "bullet_points": bullets,
        "plots": [plot1, plot2],
        "annotated_images": annotations,
        "metrics": {
            "hip_mean": float(np.nanmean(hip_s)),
            "knee_mean": float(np.nanmean(knee_s)),
            "ankle_mean": float(np.nanmean(ankle_s)),
            "trunk_mean": float(np.nanmean(trunk_s)),
            "contacts": len(contacts),
        },
    }

# ============================================================
# ANALYSE FRONTALE / VUE ARRIÈRE
# ============================================================
def run_frontal(video_path: Optional[str], patient: Dict[str, Any]) -> Dict[str, Any]:
    if not video_path:
        return default_result("Analyse frontale / vue arrière")

    frames, fps = read_video_frames(video_path, patient["sample_stride"], patient["max_frames"])
    if not frames:
        return {
            "title": "Analyse frontale / vue arrière",
            "status": "erreur",
            "summary": "Impossible de lire la vidéo.",
            "bullet_points": ["Le fichier vidéo n'a pas pu être décodé."],
            "plots": [],
            "annotated_images": [],
            "metrics": {},
        }

    mp_pl = mp.solutions.pose.PoseLandmark
    pelvis_tilt, shoulder_tilt, trunk_shift, knee_gap, heelL_y, heelR_y = [], [], [], [], [], []
    annotations = []

    for frame in frames:
        result = analyze_pose_frame(frame, static=False)
        if not result.pose_landmarks:
            pelvis_tilt.append(np.nan)
            shoulder_tilt.append(np.nan)
            trunk_shift.append(np.nan)
            knee_gap.append(np.nan)
            heelL_y.append(np.nan)
            heelR_y.append(np.nan)
            continue

        lms = result.pose_landmarks.landmark
        h, w = frame.shape[:2]
        needed = [
            mp_pl.LEFT_SHOULDER.value, mp_pl.RIGHT_SHOULDER.value,
            mp_pl.LEFT_HIP.value, mp_pl.RIGHT_HIP.value,
            mp_pl.LEFT_KNEE.value, mp_pl.RIGHT_KNEE.value,
            mp_pl.LEFT_HEEL.value, mp_pl.RIGHT_HEEL.value,
            mp_pl.NOSE.value,
        ]
        if any(vis(lms[i]) < patient["conf"] for i in needed):
            pelvis_tilt.append(np.nan)
            shoulder_tilt.append(np.nan)
            trunk_shift.append(np.nan)
            knee_gap.append(np.nan)
            heelL_y.append(np.nan)
            heelR_y.append(np.nan)
            continue

        Ls, Rs = xy(lms[mp_pl.LEFT_SHOULDER.value], w, h), xy(lms[mp_pl.RIGHT_SHOULDER.value], w, h)
        Lh, Rh = xy(lms[mp_pl.LEFT_HIP.value], w, h), xy(lms[mp_pl.RIGHT_HIP.value], w, h)
        Lk, Rk = xy(lms[mp_pl.LEFT_KNEE.value], w, h), xy(lms[mp_pl.RIGHT_KNEE.value], w, h)
        nose = xy(lms[mp_pl.NOSE.value], w, h)
        Lheel, Rheel = xy(lms[mp_pl.LEFT_HEEL.value], w, h), xy(lms[mp_pl.RIGHT_HEEL.value], w, h)

        pelvis_tilt.append(tilt_from_horizontal(Lh, Rh))
        shoulder_tilt.append(tilt_from_horizontal(Ls, Rs))
        mid_shoulders = avg_point([Ls, Rs])
        mid_pelvis = avg_point([Lh, Rh])
        trunk_shift.append(mid_shoulders[0] - mid_pelvis[0])
        knee_gap.append(abs(Lk[0] - Rk[0]))
        heelL_y.append(Lheel[1])
        heelR_y.append(Rheel[1])

    pelvis_s = smooth_ma(interp_nan(np.array(pelvis_tilt)), patient["smooth"])
    shoulder_s = smooth_ma(interp_nan(np.array(shoulder_tilt)), patient["smooth"])
    trunk_s = smooth_ma(interp_nan(np.array(trunk_shift)), patient["smooth"])
    knee_gap_s = smooth_ma(interp_nan(np.array(knee_gap)), patient["smooth"])
    heelL_s = smooth_ma(interp_nan(np.array(heelL_y)), patient["smooth"])
    heelR_s = smooth_ma(interp_nan(np.array(heelR_y)), patient["smooth"])

    contactsL, contactsR = [], []
    if len(heelL_s) >= 3:
        for i in range(1, len(heelL_s) - 1):
            if heelL_s[i] > heelL_s[i - 1] and heelL_s[i] >= heelL_s[i + 1]:
                contactsL.append(i)
            if heelR_s[i] > heelR_s[i - 1] and heelR_s[i] >= heelR_s[i + 1]:
                contactsR.append(i)

    for idx in np.linspace(0, len(frames) - 1, min(patient["num_photos"], len(frames)), dtype=int):
        r = analyze_pose_frame(frames[idx], static=False)
        img = draw_landmarks_basic(frames[idx], r.pose_landmarks, frames[idx].shape[1], frames[idx].shape[0], patient["conf"])
        out = os.path.join(tempfile.gettempdir(), f"front_annot_{idx}.png")
        cv2.imwrite(out, img)
        annotations.append(out)

    plot1 = build_plot(
        {"Bassin": pelvis_s, "Epaules": shoulder_s, "Tronc X": trunk_s},
        "Frontale_alignements",
        "Mesure (px / deg)",
        False,
    )
    plot2 = build_plot(
        {"Talon G": heelL_s, "Talon D": heelR_s},
        "Frontale_contacts",
        "Position Y (px)",
        False,
    )

    bullets = [
        f"Images analysées : {len(frames)}",
        f"Obliquité moyenne du bassin : {np.nanmean(pelvis_s):.1f} deg",
        f"Obliquité moyenne des épaules : {np.nanmean(shoulder_s):.1f} deg",
        f"Décalage latéral moyen du tronc : {np.nanmean(trunk_s):.1f} px",
        f"Distance moyenne entre genoux : {np.nanmean(knee_gap_s):.1f} px",
        f"Contacts gauche détectés : {len(contactsL)} | contacts droit détectés : {len(contactsR)}",
    ]

    return {
        "title": "Analyse frontale / vue arrière",
        "status": "ok",
        "summary": "Analyse de l'alignement frontal avec bassin, épaules, translation du tronc et détection simplifiée des contacts talon.",
        "bullet_points": bullets,
        "plots": [plot1, plot2],
        "annotated_images": annotations,
        "metrics": {
            "pelvis_mean": float(np.nanmean(pelvis_s)),
            "shoulder_mean": float(np.nanmean(shoulder_s)),
            "trunk_shift_mean": float(np.nanmean(trunk_s)),
            "contacts_left": len(contactsL),
            "contacts_right": len(contactsR),
        },
    }

# ============================================================
# ANALYSE POSTURALE FRONTALE
# ============================================================
def run_postural_front(image_path: Optional[str], patient: Dict[str, Any]) -> Dict[str, Any]:
    if not image_path:
        return default_result("Analyse posturale frontale")

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return {
            "title": "Analyse posturale frontale",
            "status": "erreur",
            "summary": "Impossible de lire l'image.",
            "bullet_points": ["Le fichier image n'a pas pu être décodé."],
            "plots": [],
            "annotated_images": [],
            "metrics": {},
        }

    img_rgb = rotate_if_landscape(ensure_rgb(img_bgr))
    result = POSE_IMAGE.process(img_rgb)
    if not result.pose_landmarks:
        return {
            "title": "Analyse posturale frontale",
            "status": "erreur",
            "summary": "Aucun squelette détecté.",
            "bullet_points": ["MediaPipe n'a pas détecté correctement la posture."],
            "plots": [],
            "annotated_images": [],
            "metrics": {},
        }

    h, w = img_rgb.shape[:2]
    lms = result.pose_landmarks.landmark
    mp_pl = mp.solutions.pose.PoseLandmark

    req = [
        mp_pl.LEFT_SHOULDER.value, mp_pl.RIGHT_SHOULDER.value,
        mp_pl.LEFT_HIP.value, mp_pl.RIGHT_HIP.value,
        mp_pl.LEFT_ANKLE.value, mp_pl.RIGHT_ANKLE.value,
        mp_pl.LEFT_EYE.value, mp_pl.RIGHT_EYE.value,
        mp_pl.NOSE.value,
    ]
    if any(vis(lms[i]) < patient["conf"] for i in req):
        return {
            "title": "Analyse posturale frontale",
            "status": "erreur",
            "summary": "Détection insuffisante.",
            "bullet_points": ["Certains points clés sont trop peu visibles pour une mesure fiable."],
            "plots": [],
            "annotated_images": [],
            "metrics": {},
        }

    Ls, Rs = xy(lms[mp_pl.LEFT_SHOULDER.value], w, h), xy(lms[mp_pl.RIGHT_SHOULDER.value], w, h)
    Lh, Rh = xy(lms[mp_pl.LEFT_HIP.value], w, h), xy(lms[mp_pl.RIGHT_HIP.value], w, h)
    La, Ra = xy(lms[mp_pl.LEFT_ANKLE.value], w, h), xy(lms[mp_pl.RIGHT_ANKLE.value], w, h)
    Le, Re = xy(lms[mp_pl.LEFT_EYE.value], w, h), xy(lms[mp_pl.RIGHT_EYE.value], w, h)
    nose = xy(lms[mp_pl.NOSE.value], w, h)

    shoulder_tilt = tilt_from_horizontal(Ls, Rs)
    pelvis_tilt = tilt_from_horizontal(Lh, Rh)
    head_tilt = tilt_from_horizontal(Le, Re)
    body_mid = avg_point([avg_point([Ls, Rs]), avg_point([Lh, Rh]), avg_point([La, Ra])])
    center_offset = nose[0] - body_mid[0]
    ankle_width = abs(La[0] - Ra[0])
    hip_width = abs(Lh[0] - Rh[0])

    ann_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ann_bgr = draw_landmarks_basic(ann_bgr, result.pose_landmarks, w, h, patient["conf"])
    ann_path = os.path.join(tempfile.gettempdir(), "posture_frontale_annotee.png")
    cv2.imwrite(ann_path, ann_bgr)

    bullets = [
        f"Inclinaison des épaules : {shoulder_tilt:.1f} deg",
        f"Inclinaison du bassin : {pelvis_tilt:.1f} deg",
        f"Inclinaison de la tête : {head_tilt:.1f} deg",
        f"Décalage horizontal de la tête par rapport à l'axe corporel : {center_offset:.1f} px",
        f"Largeur inter-chevilles : {ankle_width:.1f} px | largeur bassin : {hip_width:.1f} px",
    ]

    return {
        "title": "Analyse posturale frontale",
        "status": "ok",
        "summary": "Analyse posturale frontale réalisée à partir des repères épaules, bassin, tête et chevilles.",
        "bullet_points": bullets,
        "plots": [],
        "annotated_images": [ann_path],
        "metrics": {
            "shoulder_tilt": float(shoulder_tilt),
            "pelvis_tilt": float(pelvis_tilt),
            "head_tilt": float(head_tilt),
            "center_offset": float(center_offset),
        },
    }

# ============================================================
# ANALYSE POSTURALE LATÉRALE
# ============================================================
def run_postural_side(image_path: Optional[str], patient: Dict[str, Any]) -> Dict[str, Any]:
    if not image_path:
        return default_result("Analyse posturale latérale")

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return {
            "title": "Analyse posturale latérale",
            "status": "erreur",
            "summary": "Impossible de lire l'image.",
            "bullet_points": ["Le fichier image n'a pas pu être décodé."],
            "plots": [],
            "annotated_images": [],
            "metrics": {},
        }

    img_rgb = rotate_if_landscape(ensure_rgb(img_bgr))
    result = POSE_IMAGE.process(img_rgb)
    if not result.pose_landmarks:
        return {
            "title": "Analyse posturale latérale",
            "status": "erreur",
            "summary": "Aucun squelette détecté.",
            "bullet_points": ["MediaPipe n'a pas détecté correctement la posture."],
            "plots": [],
            "annotated_images": [],
            "metrics": {},
        }

    h, w = img_rgb.shape[:2]
    lms = result.pose_landmarks.landmark
    idx = side_indices(patient["camera_pos"])

    req = [idx["ear"], idx["shoulder"], idx["hip"], idx["knee"], idx["ankle"], idx["foot"]]
    if any(vis(lms[i]) < patient["conf"] for i in req):
        return {
            "title": "Analyse posturale latérale",
            "status": "erreur",
            "summary": "Détection insuffisante.",
            "bullet_points": ["Certains points clés sont trop peu visibles pour une mesure fiable."],
            "plots": [],
            "annotated_images": [],
            "metrics": {},
        }

    ear = xy(lms[idx["ear"]], w, h)
    shoulder = xy(lms[idx["shoulder"]], w, h)
    hip = xy(lms[idx["hip"]], w, h)
    knee = xy(lms[idx["knee"]], w, h)
    ankle = xy(lms[idx["ankle"]], w, h)
    foot = xy(lms[idx["foot"]], w, h)

    trunk_tilt = tilt_from_vertical(hip, shoulder)
    head_forward = ear[0] - shoulder[0]
    hip_angle = angle_3pts(shoulder, hip, knee)
    knee_angle = angle_3pts(hip, knee, ankle)
    ankle_angle = angle_3pts(knee, ankle, foot)

    ann_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ann_bgr = draw_landmarks_basic(ann_bgr, result.pose_landmarks, w, h, patient["conf"])
    ann_path = os.path.join(tempfile.gettempdir(), "posture_laterale_annotee.png")
    cv2.imwrite(ann_path, ann_bgr)

    bullets = [
        f"Inclinaison du tronc : {trunk_tilt:.1f} deg",
        f"Projection avant de la tête (oreille vs épaule) : {head_forward:.1f} px",
        f"Angle hanche : {hip_angle:.1f} deg",
        f"Angle genou : {knee_angle:.1f} deg",
        f"Angle cheville : {ankle_angle:.1f} deg",
    ]

    return {
        "title": "Analyse posturale latérale",
        "status": "ok",
        "summary": "Analyse posturale latérale réalisée à partir de l'alignement tête - tronc - membre inférieur.",
        "bullet_points": bullets,
        "plots": [],
        "annotated_images": [ann_path],
        "metrics": {
            "trunk_tilt": float(trunk_tilt),
            "head_forward": float(head_forward),
            "hip_angle": float(hip_angle),
            "knee_angle": float(knee_angle),
            "ankle_angle": float(ankle_angle),
        },
    }

# ============================================================
# PDF GLOBAL
# ============================================================
def build_global_pdf(patient: Dict[str, Any], results: List[Dict[str, Any]]) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)

    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, pdf_safe("Compte-rendu global biomecanique"), ln=True)
    pdf.set_font("Arial", "", 11)
    patient_name = f"{patient['nom']} {patient['prenom']}".strip() or "Non renseigne"
    pdf.cell(0, 8, pdf_safe(f"Patient : {patient_name}"), ln=True)
    pdf.cell(0, 8, pdf_safe(f"Date : {datetime.now().strftime('%d/%m/%Y %H:%M')}"), ln=True)
    pdf.cell(0, 8, pdf_safe(f"Taille : {patient['taille_cm']} cm"), ln=True)
    pdf.multi_cell(
        0,
        7,
        pdf_safe(
            f"Parametres communs - seuil visibilite : {patient['conf']} | lissage : {patient['smooth']} | "
            f"camera profil : {patient['camera_pos']} | phase : {patient['phase_cote']} | images exportees : {patient['num_photos']}"
        ),
    )
    pdf.ln(3)

    for result in results:
        pdf.set_font("Arial", "B", 13)
        pdf.cell(0, 9, pdf_safe(result["title"]), ln=True)
        pdf.set_font("Arial", "", 11)
        pdf.multi_cell(0, 7, pdf_safe(result.get("summary", "")))
        for bullet in result.get("bullet_points", []):
            pdf.multi_cell(0, 7, pdf_safe(f"- {bullet}"))
        pdf.ln(2)

        for plot in result.get("plots", [])[:2]:
            if plot and os.path.exists(plot):
                pdf.image(plot, w=180)
                pdf.ln(3)

        for img in result.get("annotated_images", [])[:2]:
            if img and os.path.exists(img):
                pdf.image(img, w=110)
                pdf.ln(3)

        pdf.ln(4)

    return bytes(pdf.output(dest="S"))

# ============================================================
# RENDU STREAMLIT
# ============================================================
def render_result(result: Dict[str, Any]) -> None:
    with st.container(border=True):
        st.subheader(result["title"])
        if result["status"] == "ok":
            st.success(result["summary"])
        elif result["status"] == "non_analyse":
            st.info(result["summary"])
        else:
            st.warning(result["summary"])

        for bullet in result.get("bullet_points", []):
            st.write(f"- {bullet}")

        if result.get("plots"):
            cols = st.columns(min(2, len(result["plots"])))
            for i, plot in enumerate(result["plots"]):
                if plot and os.path.exists(plot):
                    with cols[i % len(cols)]:
                        st.image(plot, use_container_width=True)

        if result.get("annotated_images"):
            cols = st.columns(min(3, len(result["annotated_images"])))
            for i, img in enumerate(result["annotated_images"]):
                if img and os.path.exists(img):
                    with cols[i % len(cols)]:
                        st.image(img, caption=os.path.basename(img), use_container_width=True)

# ============================================================
# EXECUTION
# ============================================================
if st.button("▶ Lancer l'analyse globale", type="primary"):
    paths = [
        save_uploaded_file(file_video_cinematique),
        save_uploaded_file(file_video_frontale),
        save_uploaded_file(file_image_frontale),
        save_uploaded_file(file_image_laterale),
    ]
    try:
        with st.spinner("Analyse en cours..."):
            results = [
                run_cinematic(paths[0], patient),
                run_frontal(paths[1], patient),
                run_postural_front(paths[2], patient),
                run_postural_side(paths[3], patient),
            ]

        st.subheader("2) Résultats")
        for result in results:
            render_result(result)

        st.subheader("3) Compte-rendu global")
        pdf_bytes = build_global_pdf(patient, results)
        filename = f"compte_rendu_global_{safe_name(patient['nom'])}_{safe_name(patient['prenom'])}.pdf"
        st.download_button("📄 Télécharger le compte-rendu PDF", data=pdf_bytes, file_name=filename, mime="application/pdf")

    finally:
        cleanup_tmp(paths)

st.markdown("---")
st.markdown(
    """
### Fonctionnement
- un seul écran pour les paramètres communs ;
- un chargement indépendant pour chaque analyse ;
- une section est marquée **Non analysé** si aucun fichier n'est fourni ;
- un seul **compte-rendu PDF global** est généré.

### Remarque
Ce code fournit une base unifiée directement exploitable.  
Les calculs reposent sur MediaPipe et sur des mesures biomécaniques simples pour rester robustes dans une seule application.
"""
)
