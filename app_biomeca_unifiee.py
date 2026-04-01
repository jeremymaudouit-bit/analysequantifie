
import io
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image
from fpdf import FPDF

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
except Exception:
    streamlit_image_coordinates = None


CV2_IMPORT_ERROR = None
MP_IMPORT_ERROR = None
try:
    import cv2
except Exception as exc:
    cv2 = None
    CV2_IMPORT_ERROR = exc

try:
    import mediapipe as mp
except Exception as exc:
    mp = None
    MP_IMPORT_ERROR = exc

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="Biomeca Suite - Analyse unifiée", layout="wide")
st.title("🧍🏃 Biomeca Suite - Analyse unifiée")
st.caption("Une seule application pour l'analyse cinématique, frontale et posturale avec un rapport global unique.")

FPS_DEFAULT = 30
POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS if mp is not None else None

# ============================================================
# MEDIAPIPE
# ============================================================
@st.cache_resource
def load_video_pose():
    if mp is None:
        return None
    return mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

@st.cache_resource
def load_image_pose():
    if mp is None:
        return None
    return mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

POSE_VIDEO = load_video_pose()
POSE_IMAGE = load_image_pose()

def dependencies_ready() -> bool:
    return cv2 is not None and mp is not None and POSE_VIDEO is not None and POSE_IMAGE is not None

def show_dependency_help() -> None:
    st.error("Les dépendances OpenCV / MediaPipe ne sont pas correctement chargées sur cet environnement.")
    if CV2_IMPORT_ERROR is not None:
        st.code(f"Erreur cv2: {CV2_IMPORT_ERROR}")
    if MP_IMPORT_ERROR is not None:
        st.code(f"Erreur mediapipe: {MP_IMPORT_ERROR}")
    st.markdown(
        """
### Correctif recommandé
Utilisez ces versions dans `requirements.txt` :
```txt
streamlit==1.44.1
numpy<2
opencv-python-headless==4.10.0.84
mediapipe==0.10.21
matplotlib==3.8.4
Pillow==10.4.0
fpdf2==2.8.3
```
Et pour l'hébergement, gardez `runtime.txt` sur Python 3.11.
"""
    )

# ============================================================
# PARAMÈTRES COMMUNS
# ============================================================
if "override_front_points" not in st.session_state:
    st.session_state["override_front_points"] = {}
if "override_side_points" not in st.session_state:
    st.session_state["override_side_points"] = {}

with st.sidebar:
    st.header("Paramètres communs")
    nom = st.text_input("Nom", "")
    prenom = st.text_input("Prénom", "")
    taille_cm = st.number_input("Taille du patient (cm)", min_value=80, max_value=230, value=170, step=1)

    st.divider()
    st.subheader("Analyses dynamiques")
    conf = st.slider("Seuil de visibilité minimal", 0.1, 0.95, 0.5, 0.05)
    smooth = st.slider("Lissage", 0, 10, 3, 1)
    show_norm = st.checkbox("Afficher les courbes indicatives de référence", value=True)
    camera_pos = st.selectbox("Côté / angle vidéo de profil", ["Profil droit", "Profil gauche"])
    phase_cote = st.selectbox("Phases / côté dominant", ["Aucune", "Droite", "Gauche", "Les deux"])
    sample_stride = st.slider("Pas d'échantillonnage vidéo (1 = toutes les images)", 1, 6, 2)
    num_photos = st.slider("Nombre d'images annotées à exporter", 1, 8, 3)
    max_frames = st.slider("Nombre maximal d'images analysées par vidéo", 60, 600, 180, 30)

    st.divider()
    st.subheader("Correction statique par clic")
    click_edit_enabled = st.checkbox(
        "Activer la correction des points statiques",
        value=True,
        help="Permet de déplacer manuellement certains points avant le calcul, comme dans les applications statiques d'origine.",
    )
    display_width = st.slider("Largeur d'affichage image (px)", 320, 900, 520, 10)
    auto_crop_static = st.checkbox("Cadrage automatique des images statiques", value=True)

    editable_front_points = [
        "Hanche G", "Hanche D", "Genou G", "Genou D",
        "Cheville G", "Cheville D", "Talon G", "Talon D"
    ]
    editable_side_points = ["Epaule", "Hanche", "Genou", "Cheville", "Talon", "Oreille", "Nez"]

    point_to_edit_front = st.selectbox(
        "Point frontal à corriger",
        editable_front_points,
        disabled=not click_edit_enabled,
    )
    point_to_edit_side = st.selectbox(
        "Point latéral à corriger",
        editable_side_points,
        disabled=not click_edit_enabled,
    )

    c_reset1, c_reset2 = st.columns(2)
    with c_reset1:
        if st.button("Reset point frontal", disabled=not click_edit_enabled):
            st.session_state["override_front_points"].pop(point_to_edit_front, None)
    with c_reset2:
        if st.button("Reset point latéral", disabled=not click_edit_enabled):
            st.session_state["override_side_points"].pop(point_to_edit_side, None)

    c_reset3, c_reset4 = st.columns(2)
    with c_reset3:
        if st.button("Reset frontal", disabled=not click_edit_enabled):
            st.session_state["override_front_points"] = {}
    with c_reset4:
        if st.button("Reset latéral", disabled=not click_edit_enabled):
            st.session_state["override_side_points"] = {}

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
    "click_edit_enabled": bool(click_edit_enabled),
    "display_width": int(display_width),
    "auto_crop_static": bool(auto_crop_static),
    "point_to_edit_front": point_to_edit_front,
    "point_to_edit_side": point_to_edit_side,
}

if not dependencies_ready():
    show_dependency_help()
    st.stop()

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

    mp_pl = mp.solutions.pose.PoseLandmark
    res = {k: [] for k in ["Tronc", "Hanche G", "Hanche D", "Genou G", "Genou D", "Cheville G", "Cheville D"]}
    heelG_y, heelD_y, heelG_x, heelD_x, toeG_x, toeD_x = [], [], [], [], [], []
    annotations = []

    for frame in frames:
        result = analyze_pose_frame(frame, static=False)
        if not result.pose_landmarks:
            for k in res:
                res[k].append(np.nan)
            heelG_y.append(np.nan); heelD_y.append(np.nan)
            heelG_x.append(np.nan); heelD_x.append(np.nan)
            toeG_x.append(np.nan); toeD_x.append(np.nan)
            continue

        lms = result.pose_landmarks.landmark
        def pt(enum_):
            p = lms[enum_.value]
            return np.array([p.x, p.y], dtype=np.float32), float(p.visibility)
        kp = {}
        for side, suf in [("LEFT", "G"), ("RIGHT", "D")]:
            kp[f"Epaule {suf}"], kp[f"Epaule {suf} vis"] = pt(getattr(mp_pl, f"{side}_SHOULDER"))
            kp[f"Hanche {suf}"], kp[f"Hanche {suf} vis"] = pt(getattr(mp_pl, f"{side}_HIP"))
            kp[f"Genou {suf}"], kp[f"Genou {suf} vis"] = pt(getattr(mp_pl, f"{side}_KNEE"))
            kp[f"Cheville {suf}"], kp[f"Cheville {suf} vis"] = pt(getattr(mp_pl, f"{side}_ANKLE"))
            kp[f"Talon {suf}"], kp[f"Talon {suf} vis"] = pt(getattr(mp_pl, f"{side}_HEEL"))
            kp[f"Orteil {suf}"], kp[f"Orteil {suf} vis"] = pt(getattr(mp_pl, f"{side}_FOOT_INDEX"))

        def ok(name):
            return kp.get(f"{name} vis", 0.0) >= patient["conf"]

        res["Tronc"].append(
            angle_tronc_frontal(kp["Epaule G"][::-1], kp["Epaule D"][::-1], kp["Hanche G"][::-1], kp["Hanche D"][::-1]) * 0
            if False else (
                float(np.degrees(np.arctan2(
                    ((kp["Epaule G"][0] + kp["Epaule D"][0]) / 2.0) - ((kp["Hanche G"][0] + kp["Hanche D"][0]) / 2.0),
                    -(((kp["Epaule G"][1] + kp["Epaule D"][1]) / 2.0) - ((kp["Hanche G"][1] + kp["Hanche D"][1]) / 2.0)) + 1e-6
                )))
                if (ok("Epaule G") and ok("Epaule D") and ok("Hanche G") and ok("Hanche D")) else np.nan
            )
        )
        res["Hanche G"].append(angle_hanche(kp["Epaule G"][0:2], kp["Hanche G"][0:2], kp["Genou G"][0:2]) if (ok("Epaule G") and ok("Hanche G") and ok("Genou G")) else np.nan)
        res["Hanche D"].append(angle_hanche(kp["Epaule D"][0:2], kp["Hanche D"][0:2], kp["Genou D"][0:2]) if (ok("Epaule D") and ok("Hanche D") and ok("Genou D")) else np.nan)
        res["Genou G"].append(angle_genou(kp["Hanche G"][0:2], kp["Genou G"][0:2], kp["Cheville G"][0:2]) if (ok("Hanche G") and ok("Genou G") and ok("Cheville G")) else np.nan)
        res["Genou D"].append(angle_genou(kp["Hanche D"][0:2], kp["Genou D"][0:2], kp["Cheville D"][0:2]) if (ok("Hanche D") and ok("Genou D") and ok("Cheville D")) else np.nan)
        res["Cheville G"].append(angle_cheville(kp["Genou G"][0:2], kp["Cheville G"][0:2], kp["Talon G"][0:2], kp["Orteil G"][0:2]) if (ok("Genou G") and ok("Cheville G") and ok("Talon G") and ok("Orteil G")) else np.nan)
        res["Cheville D"].append(angle_cheville(kp["Genou D"][0:2], kp["Cheville D"][0:2], kp["Talon D"][0:2], kp["Orteil D"][0:2]) if (ok("Genou D") and ok("Cheville D") and ok("Talon D") and ok("Orteil D")) else np.nan)

        heelG_y.append(float(kp["Talon G"][1]) if ok("Talon G") else np.nan)
        heelD_y.append(float(kp["Talon D"][1]) if ok("Talon D") else np.nan)
        heelG_x.append(float(kp["Talon G"][0]) if ok("Talon G") else np.nan)
        heelD_x.append(float(kp["Talon D"][0]) if ok("Talon D") else np.nan)
        toeG_x.append(float(kp["Orteil G"][0]) if ok("Orteil G") else np.nan)
        toeD_x.append(float(kp["Orteil D"][0]) if ok("Orteil D") else np.nan)

    contactsG, heelG_s = detect_foot_contacts(heelG_y, fps=fps)
    contactsD, heelD_s = detect_foot_contacts(heelD_y, fps=fps)
    _, step_time_G_mean, step_time_G_std = compute_step_times(contactsG, fps=fps)
    _, step_time_D_mean, step_time_D_std = compute_step_times(contactsD, fps=fps)

    phases = []
    if patient["phase_cote"] in ["Gauche", "Les deux"]:
        c = detect_cycle(heelG_y)
        if c: phases.append((*c, "orange"))
    if patient["phase_cote"] in ["Droite", "Les deux"]:
        c = detect_cycle(heelD_y)
        if c: phases.append((*c, "blue"))

    figures = []
    table_metrics = []
    asym_rows = []
    # Tronc
    trunk_raw = np.array(res["Tronc"], dtype=float)
    trunk = smooth_clinical(trunk_raw, smooth_level=patient["smooth"])
    figures.append(build_plot({"Tronc": trunk}, "Cinematique_Tronc", "Angle (deg)", patient["show_norm"]))
    if np.sum(~np.isnan(trunk_raw)):
        vals = trunk[~np.isnan(trunk_raw)]
        table_metrics.append(("Tronc", float(np.min(vals)), float(np.mean(vals)), float(np.max(vals))))
    # articulations
    for joint in ["Hanche", "Genou", "Cheville"]:
        g_raw = np.array(res[f"{joint} G"], dtype=float)
        d_raw = np.array(res[f"{joint} D"], dtype=float)
        g = smooth_clinical(g_raw, smooth_level=patient["smooth"])
        d = smooth_clinical(d_raw, smooth_level=patient["smooth"])
        figures.append(build_plot({f"{joint} G": g, f"{joint} D": d}, f"Cinematique_{joint}", "Angle (deg)", patient["show_norm"]))
        for side, arrf, arrr in [("Gauche", g, g_raw), ("Droite", d, d_raw)]:
            mask = ~np.isnan(arrr)
            if mask.sum():
                vals = arrf[mask]
                table_metrics.append((f"{joint} {side}", float(np.min(vals)), float(np.mean(vals)), float(np.max(vals))))
        gmean = float(np.mean(g[~np.isnan(g_raw)])) if np.sum(~np.isnan(g_raw)) else None
        dmean = float(np.mean(d[~np.isnan(d_raw)])) if np.sum(~np.isnan(d_raw)) else None
        asym = None if (gmean is None or dmean is None or abs((gmean+dmean)/2.0) < 1e-6) else 100.0 * abs(dmean-gmean) / abs((gmean+dmean)/2.0)
        asym_rows.append((joint, gmean, dmean, asym))

    for idx in np.linspace(0, len(frames) - 1, min(patient["num_photos"], len(frames)), dtype=int):
        r = analyze_pose_frame(frames[idx], static=False)
        img = draw_landmarks_basic(frames[idx], r.pose_landmarks, frames[idx].shape[1], frames[idx].shape[0], patient["conf"])
        out = os.path.join(tempfile.gettempdir(), f"cin_annot_{idx}.png")
        cv2.imwrite(out, img)
        annotations.append(out)

    bullets = []
    if table_metrics:
        for label, vmin, vmean, vmax in table_metrics[:7]:
            bullets.append(f"{label} - min {vmin:.1f} deg | moyenne {vmean:.1f} deg | max {vmax:.1f} deg")
    for joint, gmean, dmean, asym in asym_rows:
        if gmean is not None and dmean is not None:
            if asym is None:
                bullets.append(f"{joint} - moy G {gmean:.1f} deg | moy D {dmean:.1f} deg")
            else:
                bullets.append(f"{joint} - moy G {gmean:.1f} deg | moy D {dmean:.1f} deg | asym {asym:.1f} %")
    if step_time_G_mean is not None:
        bullets.append(f"Temps du pas gauche : {step_time_G_mean:.2f} s (+/- {step_time_G_std:.2f} s)")
    else:
        bullets.append("Temps du pas gauche non calculable")
    if step_time_D_mean is not None:
        bullets.append(f"Temps du pas droit : {step_time_D_mean:.2f} s (+/- {step_time_D_std:.2f} s)")
    else:
        bullets.append("Temps du pas droit non calculable")
    bullets.append(f"Contacts detectes : gauche {len(contactsG)} | droit {len(contactsD)}")

    return {
        "title": "Analyse cinématique",
        "status": "ok",
        "summary": "Analyse cinématique calculée avec les mêmes angles principaux que dans l'application d'origine : tronc, hanche, genou et cheville.",
        "bullet_points": bullets,
        "plots": figures,
        "annotated_images": annotations,
        "metrics": {"contactsG": len(contactsG), "contactsD": len(contactsD)},
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
    data = {"Genou G": [], "Genou D": [], "Arriere-pied G": [], "Arriere-pied D": [], "Bassin": [], "Tronc": []}
    heelG_y, heelD_y = [], []
    annotations = []

    for frame in frames:
        result = analyze_pose_frame(frame, static=False)
        if not result.pose_landmarks:
            for k in data: data[k].append(np.nan)
            heelG_y.append(np.nan); heelD_y.append(np.nan)
            continue
        lms = result.pose_landmarks.landmark
        def pt(enum_):
            p = lms[enum_.value]
            return np.array([p.x, p.y], dtype=np.float32), float(p.visibility)
        kp = {}
        for side, suf in [("LEFT", "G"), ("RIGHT", "D")]:
            kp[f"Epaule {suf}"], kp[f"Epaule {suf} vis"] = pt(getattr(mp_pl, f"{side}_SHOULDER"))
            kp[f"Hanche {suf}"], kp[f"Hanche {suf} vis"] = pt(getattr(mp_pl, f"{side}_HIP"))
            kp[f"Genou {suf}"], kp[f"Genou {suf} vis"] = pt(getattr(mp_pl, f"{side}_KNEE"))
            kp[f"Cheville {suf}"], kp[f"Cheville {suf} vis"] = pt(getattr(mp_pl, f"{side}_ANKLE"))
            kp[f"Talon {suf}"], kp[f"Talon {suf} vis"] = pt(getattr(mp_pl, f"{side}_HEEL"))
        def ok(name):
            return kp.get(f"{name} vis", 0.0) >= patient["conf"]

        data["Genou G"].append(angle_genou_frontal(kp["Hanche G"][0:2], kp["Genou G"][0:2], kp["Cheville G"][0:2], side="G") if (ok("Hanche G") and ok("Genou G") and ok("Cheville G")) else np.nan)
        data["Genou D"].append(angle_genou_frontal(kp["Hanche D"][0:2], kp["Genou D"][0:2], kp["Cheville D"][0:2], side="D") if (ok("Hanche D") and ok("Genou D") and ok("Cheville D")) else np.nan)
        data["Arriere-pied G"].append(angle_arriere_pied_frontal(kp["Cheville G"][0:2], kp["Talon G"][0:2], side="G") if (ok("Cheville G") and ok("Talon G")) else np.nan)
        data["Arriere-pied D"].append(angle_arriere_pied_frontal(kp["Cheville D"][0:2], kp["Talon D"][0:2], side="D") if (ok("Cheville D") and ok("Talon D")) else np.nan)
        data["Bassin"].append(angle_bassin_frontal(kp["Hanche G"][0:2], kp["Hanche D"][0:2]) if (ok("Hanche G") and ok("Hanche D")) else np.nan)
        data["Tronc"].append(angle_tronc_frontal(kp["Epaule G"][0:2], kp["Epaule D"][0:2], kp["Hanche G"][0:2], kp["Hanche D"][0:2]) if (ok("Epaule G") and ok("Epaule D") and ok("Hanche G") and ok("Hanche D")) else np.nan)
        heelG_y.append(float(kp["Talon G"][1]) if ok("Talon G") else np.nan)
        heelD_y.append(float(kp["Talon D"][1]) if ok("Talon D") else np.nan)

    contactsG, heelG_s = detect_foot_contacts(heelG_y, fps=fps)
    contactsD, heelD_s = detect_foot_contacts(heelD_y, fps=fps)
    _, step_time_G_mean, step_time_G_std = compute_step_times(contactsG, fps=fps)
    _, step_time_D_mean, step_time_D_std = compute_step_times(contactsD, fps=fps)

    phases = []
    if patient["phase_cote"] in ["Gauche", "Les deux"]:
        c = detect_cycle(heelG_y)
        if c: phases.append((*c, "orange"))
    if patient["phase_cote"] in ["Droite", "Les deux"]:
        c = detect_cycle(heelD_y)
        if c: phases.append((*c, "blue"))

    figures = []
    table_metrics = []
    asym_rows = []
    for label, lk, rk in [("Genou", "Genou G", "Genou D"), ("Arriere-pied", "Arriere-pied G", "Arriere-pied D")]:
        g_raw = np.array(data[lk], dtype=float)
        d_raw = np.array(data[rk], dtype=float)
        local_smooth = patient["smooth"] + 2 if label == "Arriere-pied" else patient["smooth"]
        g = smooth_clinical(g_raw, smooth_level=local_smooth)
        d = smooth_clinical(d_raw, smooth_level=local_smooth)
        figures.append(build_plot({lk: g, rk: d}, f"Frontale_{label}", "Angle (deg)", patient["show_norm"]))
        for side, arrf, arrr in [("Gauche", g, g_raw), ("Droite", d, d_raw)]:
            mask = ~np.isnan(arrr)
            if mask.sum():
                vals = arrf[mask]
                table_metrics.append((f"{label} {side}", float(np.min(vals)), float(np.mean(vals)), float(np.max(vals))))
        gmean = float(np.mean(g[~np.isnan(g_raw)])) if np.sum(~np.isnan(g_raw)) else None
        dmean = float(np.mean(d[~np.isnan(d_raw)])) if np.sum(~np.isnan(d_raw)) else None
        asym = None if (gmean is None or dmean is None or abs((gmean+dmean)/2.0) < 1e-6) else 100.0 * abs(dmean-gmean) / abs((gmean+dmean)/2.0)
        asym_rows.append((label, gmean, dmean, asym))

    for label in ["Bassin", "Tronc"]:
        raw = np.array(data[label], dtype=float)
        val = smooth_clinical(raw, smooth_level=patient["smooth"])
        figures.append(build_plot({label: val}, f"Frontale_{label}", "Angle (deg)", patient["show_norm"]))
        mask = ~np.isnan(raw)
        if mask.sum():
            vals = val[mask]
            table_metrics.append((label, float(np.min(vals)), float(np.mean(vals)), float(np.max(vals))))

    figures.append(build_plot({"Talon G": heelG_s, "Talon D": heelD_s}, "Frontale_contacts", "Position Y (px)", False))

    for idx in np.linspace(0, len(frames) - 1, min(patient["num_photos"], len(frames)), dtype=int):
        r = analyze_pose_frame(frames[idx], static=False)
        img = draw_landmarks_basic(frames[idx], r.pose_landmarks, frames[idx].shape[1], frames[idx].shape[0], patient["conf"])
        out = os.path.join(tempfile.gettempdir(), f"front_annot_{idx}.png")
        cv2.imwrite(out, img)
        annotations.append(out)

    bullets = []
    for label, vmin, vmean, vmax in table_metrics[:8]:
        bullets.append(f"{label} - min {vmin:.1f} deg | moyenne {vmean:.1f} deg | max {vmax:.1f} deg")
    for joint, gmean, dmean, asym in asym_rows:
        if gmean is not None and dmean is not None:
            if asym is None:
                bullets.append(f"{joint} - moy G {gmean:.1f} deg | moy D {dmean:.1f} deg")
            else:
                bullets.append(f"{joint} - moy G {gmean:.1f} deg | moy D {dmean:.1f} deg | asym {asym:.1f} %")
    if step_time_G_mean is not None:
        bullets.append(f"Temps du pas gauche : {step_time_G_mean:.2f} s (+/- {step_time_G_std:.2f} s)")
    else:
        bullets.append("Temps du pas gauche non calculable")
    if step_time_D_mean is not None:
        bullets.append(f"Temps du pas droit : {step_time_D_mean:.2f} s (+/- {step_time_D_std:.2f} s)")
    else:
        bullets.append("Temps du pas droit non calculable")
    bullets.append(f"Contacts detectes : gauche {len(contactsG)} | droit {len(contactsD)}")

    return {
        "title": "Analyse frontale / vue arrière",
        "status": "ok",
        "summary": "Analyse frontale recalée sur les mesures de l'application d'origine : genoux, arrière-pied, bassin et tronc.",
        "bullet_points": bullets,
        "plots": figures,
        "annotated_images": annotations,
        "metrics": {"contactsG": len(contactsG), "contactsD": len(contactsD)},
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
    res_for_crop = POSE_IMAGE.process(img_rgb)
    if patient.get("auto_crop_static"):
        img_rgb = crop_to_landmarks(img_rgb, res_for_crop, pad_ratio=0.18)

    origin_points = extract_origin_points_from_mediapipe_front(img_rgb)
    if origin_points is None:
        return {
            "title": "Analyse posturale frontale",
            "status": "erreur",
            "summary": "Aucune pose détectée.",
            "bullet_points": ["Utilisez une photo nette, en pied, bien centrée."],
            "plots": [],
            "annotated_images": [],
            "metrics": {},
        }

    if patient.get("click_edit_enabled") and streamlit_image_coordinates is not None:
        st.markdown("#### Correction de points - posture frontale")
        disp_w = min(int(patient["display_width"]), img_rgb.shape[1])
        scale = disp_w / img_rgb.shape[1]
        disp_h = int(img_rgb.shape[0] * scale)
        img_disp = cv2.resize(img_rgb, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        preview = draw_preview(img_disp, origin_points, st.session_state["override_front_points"], scale)
        coords = streamlit_image_coordinates(Image.open(io.BytesIO(to_png_bytes(preview))), key="front_click")
        if coords is not None:
            x_orig = float(coords["x"]) / scale
            y_orig = float(coords["y"]) / scale
            st.session_state["override_front_points"][patient["point_to_edit_front"]] = (x_orig, y_orig)
            st.success(f"Point frontal {patient['point_to_edit_front']} placé à ({x_orig:.0f}, {y_orig:.0f}) px")

    points = {k: np.array(v, dtype=np.float32) for k, v in origin_points.items()}
    for k, v in st.session_state["override_front_points"].items():
        if k in points:
            points[k] = np.array([v[0], v[1]], dtype=np.float32)

    LS, RS = points["Hanche G"]*0 + origin_points["Epaule G"], points["Hanche D"]*0 + origin_points["Epaule D"]
    LH, RH = points["Hanche G"], points["Hanche D"]
    LK, RK = points["Genou G"], points["Genou D"]
    LA, RA = points["Cheville G"], points["Cheville D"]
    LHeel, RHeel = points["Talon G"], points["Talon D"]

    shoulder_tilt = tilt_from_horizontal(LS, RS)
    pelvis_tilt = tilt_from_horizontal(LH, RH)
    knee_g = angle_genou_frontal(LH, LK, LA, side="G")
    knee_d = angle_genou_frontal(RH, RK, RA, side="D")
    rear_g = angle_arriere_pied_frontal(LA, LHeel, side="G")
    rear_d = angle_arriere_pied_frontal(RA, RHeel, side="D")
    hip_width = abs(LH[0] - RH[0])
    ankle_width = abs(LA[0] - RA[0])

    ann_bgr = cv2.cvtColor(img_rgb.copy(), cv2.COLOR_RGB2BGR)
    for _, p in points.items():
        cv2.circle(ann_bgr, tuple(np.round(p).astype(int)), 7, (0, 255, 0), -1)
    for name, p in st.session_state["override_front_points"].items():
        arr = np.array(p)
        cv2.circle(ann_bgr, tuple(np.round(arr).astype(int)), 14, (255, 0, 255), 3)
        cv2.putText(ann_bgr, name, (int(arr[0]) + 8, int(arr[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    ann_path = os.path.join(tempfile.gettempdir(), "posture_frontale_annotee.png")
    cv2.imwrite(ann_path, ann_bgr)

    bullets = [
        f"Inclinaison epaules : {shoulder_tilt:.1f} deg",
        f"Inclinaison bassin : {pelvis_tilt:.1f} deg",
        f"Genou G : {knee_g:.1f} deg | Genou D : {knee_d:.1f} deg",
        f"Arriere-pied G : {rear_g:.1f} deg | Arriere-pied D : {rear_d:.1f} deg",
        f"Largeur bassin : {hip_width:.1f} px | largeur inter-chevilles : {ankle_width:.1f} px",
    ]
    if st.session_state["override_front_points"]:
        bullets.append(f"Points corriges manuellement : {', '.join(st.session_state['override_front_points'].keys())}")

    return {
        "title": "Analyse posturale frontale",
        "status": "ok",
        "summary": "Analyse statique frontale avec possibilité de correction manuelle des points avant calcul.",
        "bullet_points": bullets,
        "plots": [],
        "annotated_images": [ann_path],
        "metrics": {"shoulder_tilt": float(shoulder_tilt), "pelvis_tilt": float(pelvis_tilt), "knee_g": float(knee_g), "knee_d": float(knee_d)},
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
    res_for_crop = POSE_IMAGE.process(img_rgb)
    if patient.get("auto_crop_static"):
        img_rgb = crop_to_landmarks(img_rgb, res_for_crop, pad_ratio=0.18)

    side_detected, origin_points = extract_points_side(img_rgb)
    if origin_points is None:
        return {
            "title": "Analyse posturale latérale",
            "status": "erreur",
            "summary": "Aucune pose détectée.",
            "bullet_points": ["Utilisez une photo nette, de profil, en pied."],
            "plots": [],
            "annotated_images": [],
            "metrics": {},
        }

    if patient.get("click_edit_enabled") and streamlit_image_coordinates is not None:
        st.markdown("#### Correction de points - posture latérale")
        st.caption(f"Côté détecté : {'Gauche' if side_detected == 'left' else 'Droite'}")
        disp_w = min(int(patient["display_width"]), img_rgb.shape[1])
        scale = disp_w / img_rgb.shape[1]
        disp_h = int(img_rgb.shape[0] * scale)
        img_disp = cv2.resize(img_rgb, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        preview = draw_preview(img_disp, origin_points, st.session_state["override_side_points"], scale)
        coords = streamlit_image_coordinates(Image.open(io.BytesIO(to_png_bytes(preview))), key="side_click")
        if coords is not None:
            x_orig = float(coords["x"]) / scale
            y_orig = float(coords["y"]) / scale
            st.session_state["override_side_points"][patient["point_to_edit_side"]] = (x_orig, y_orig)
            st.success(f"Point latéral {patient['point_to_edit_side']} placé à ({x_orig:.0f}, {y_orig:.0f}) px")

    points = {k: np.array(v, dtype=np.float32) for k, v in origin_points.items()}
    for k, v in st.session_state["override_side_points"].items():
        if k in points:
            points[k] = np.array([v[0], v[1]], dtype=np.float32)

    Epaule = points["Epaule"]; Hanche = points["Hanche"]; Genou = points["Genou"]
    Cheville = points["Cheville"]; Talon = points["Talon"]; Oreille = points["Oreille"]; Nez = points["Nez"]

    incl_jambe = abs(signed_angle_vs_vertical(Genou, Cheville))
    incl_cuisse = abs(signed_angle_vs_vertical(Hanche, Genou))
    incl_tronc = abs(signed_angle_vs_vertical(Hanche, Epaule))
    incl_tete_cou = abs(signed_angle_vs_vertical(Epaule, Oreille))
    incl_tete_nez = abs(signed_angle_vs_vertical(Oreille, Nez))
    signed_tronc = signed_angle_vs_vertical(Hanche, Epaule)
    signed_tete = signed_angle_vs_vertical(Epaule, Oreille)
    angle_gen = angle( Hanche, Genou, Cheville)
    angle_chev = angle(Genou, Cheville, Talon)
    sens_tronc = "Vers l'avant" if signed_tronc > 0 else "Vers l'arriere"
    sens_tete = "Vers l'avant" if signed_tete > 0 else "Vers l'arriere"

    ann_bgr = cv2.cvtColor(img_rgb.copy(), cv2.COLOR_RGB2BGR)
    for _, p in points.items():
        cv2.circle(ann_bgr, tuple(np.round(p).astype(int)), 7, (0, 255, 0), -1)
    for name, p in st.session_state["override_side_points"].items():
        arr = np.array(p)
        cv2.circle(ann_bgr, tuple(np.round(arr).astype(int)), 14, (255, 0, 255), 3)
        cv2.putText(ann_bgr, name, (int(arr[0]) + 8, int(arr[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    for a, b, color in [(Hanche, Epaule, (255,0,0)), (Hanche, Genou, (0,200,0)), (Genou, Cheville, (0,255,255)), (Epaule, Oreille, (255,0,255)), (Oreille, Nez, (100,255,100))]:
        cv2.line(ann_bgr, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), color, 3)
    ann_path = os.path.join(tempfile.gettempdir(), "posture_laterale_annotee.png")
    cv2.imwrite(ann_path, ann_bgr)

    bullets = [
        f"Cote detecte : {'Gauche' if side_detected == 'left' else 'Droite'}",
        f"Inclinaison jambe / verticale : {incl_jambe:.1f} deg",
        f"Inclinaison cuisse / verticale : {incl_cuisse:.1f} deg",
        f"Inclinaison tronc / verticale : {incl_tronc:.1f} deg ({sens_tronc})",
        f"Inclinaison tete-cou / verticale : {incl_tete_cou:.1f} deg ({sens_tete})",
        f"Inclinaison tete (oreille-nez) : {incl_tete_nez:.1f} deg",
        f"Angle genou : {angle_gen:.1f} deg | Angle cheville : {angle_chev:.1f} deg",
    ]
    if st.session_state["override_side_points"]:
        bullets.append(f"Points corriges manuellement : {', '.join(st.session_state['override_side_points'].keys())}")

    return {
        "title": "Analyse posturale latérale",
        "status": "ok",
        "summary": "Analyse statique latérale avec les mêmes angles de référence que l'application d'origine et correction manuelle possible.",
        "bullet_points": bullets,
        "plots": [],
        "annotated_images": [ann_path],
        "metrics": {"incl_tronc": float(incl_tronc), "angle_genou": float(angle_gen), "angle_cheville": float(angle_chev)},
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
    pdf_multicell_safe(pdf, f"Patient : {patient_name}")
    pdf_multicell_safe(pdf, f"Date : {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    pdf_multicell_safe(pdf, f"Taille : {patient['taille_cm']} cm")
    pdf_multicell_safe(pdf, f"Parametres communs - seuil visibilite : {patient['conf']} | lissage : {patient['smooth']} | camera profil : {patient['camera_pos']} | phase : {patient['phase_cote']} | images exportees : {patient['num_photos']}")
    pdf.ln(2)
    for result in results:
        pdf.set_font("Arial", "B", 13)
        pdf_multicell_safe(pdf, result["title"])
        pdf.set_font("Arial", "", 11)
        pdf_multicell_safe(pdf, result.get("summary", ""))
        bullets = result.get("bullet_points", []) or ["Non analyse"]
        for bullet in bullets:
            pdf_multicell_safe(pdf, f"- {bullet}")
        pdf.ln(2)
        for plot in result.get("plots", [])[:2]:
            if plot and os.path.exists(plot):
                try:
                    pdf.image(plot, w=180)
                    pdf.ln(3)
                except Exception:
                    pass
        for img in result.get("annotated_images", [])[:2]:
            if img and os.path.exists(img):
                try:
                    pdf.image(img, w=110)
                    pdf.ln(3)
                except Exception:
                    pass
        pdf.ln(4)
    out = pdf.output(dest="S")
    if isinstance(out, bytearray):
        return bytes(out)
    if isinstance(out, str):
        return out.encode("latin-1", errors="ignore")
    return bytes(out)

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
