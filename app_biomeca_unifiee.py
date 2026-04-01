# -*- coding: utf-8 -*-
"""
Version monolithique explicite.
- 1 seule saisie des paramètres communs
- mêmes codes d'origine conservés dans 4 blocs explicites
- aucune réécriture des calculs métier dans les blocs legacy
"""

from __future__ import annotations
import io
import os
import re
import sys
import types
import traceback
from datetime import datetime
from contextlib import contextmanager
from typing import Any, Dict, List

import streamlit as st
import streamlit.components.v1 as components
from fpdf import FPDF


# ============================================================
# BLOCS ORIGINAUX CONSERVÉS
# ============================================================
FRONTALE_CODE = r"""
import streamlit as st
import cv2, os, tempfile, base64
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.signal import find_peaks
import mediapipe as mp

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Image as PDFImage,
    Spacer, Table, TableStyle, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors

import streamlit.components.v1 as components

# ==============================
# CONFIG
# ==============================
st.set_page_config("GaitScan Pro (Analyse frontale)", layout="wide")
st.title("🏃 GaitScan Pro – Analyse Frontale / Vue arrière")
FPS = 30

# ==============================
# MEDIAPIPE
# ==============================
mp_pose = mp.solutions.pose

@st.cache_resource
def load_pose():
    return mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

pose = load_pose()

# ==============================
# NORMES INDICATIVES
# ==============================
def norm_curve(metric, n):
    x = np.linspace(0, 100, n)

    if metric in ["Genou G", "Genou D"]:
        return np.interp(x, [0, 25, 50, 75, 100], [1, 4, 0, -4, 1])

    if metric in ["Arriere-pied G", "Arriere-pied D"]:
        return np.interp(x, [0, 25, 50, 75, 100], [0.5, 2, 0, -2, 0.5])

    if metric == "Bassin":
        return np.interp(x, [0, 25, 50, 75, 100], [0, 2, 0, -2, 0])

    if metric == "Tronc":
        return np.interp(x, [0, 25, 50, 75, 100], [0, 2, 0, -2, 0])

    return np.zeros(n)

def smooth_ma(y, win=7):
    y = np.asarray(y, dtype=float)
    if win is None or win <= 1:
        return y
    win = int(win)
    if win % 2 == 0:
        win += 1
    pad = win // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(win, dtype=float) / win
    return np.convolve(ypad, kernel, mode="valid")

# ==============================
# OUTLIERS + LISSAGE
# ==============================
def interp_nan(arr):
    arr = np.asarray(arr, dtype=float)
    idx = np.arange(len(arr))
    ok = ~np.isnan(arr)
    if ok.sum() >= 2:
        return np.interp(idx, idx[ok], arr[ok])
    return np.zeros_like(arr)

def remove_outliers_hampel(x, win=5, n_sigmas=3.0):
    x = np.asarray(x, dtype=float).copy()
    n = len(x)
    if n < 3:
        return x

    y = x.copy()
    k = 1.4826

    for i in range(n):
        i0 = max(0, i - win)
        i1 = min(n, i + win + 1)
        w = x[i0:i1]
        med = np.median(w)
        mad = np.median(np.abs(w - med))

        if mad < 1e-9:
            continue

        if abs(x[i] - med) > n_sigmas * k * mad:
            y[i] = med

    return y

def smooth_clinical(arr, smooth_level=3):
    x = interp_nan(arr)
    x = remove_outliers_hampel(x, win=3 + smooth_level, n_sigmas=3.0)
    win = max(3, 2 * smooth_level + 3)
    if win % 2 == 0:
        win += 1
    return smooth_ma(x, win=win)

# ==============================
# POSE DETECTION
# ==============================
def detect_pose(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(img_rgb)
    if not res.pose_landmarks:
        return None

    lm = res.pose_landmarks.landmark
    L = mp_pose.PoseLandmark

    def pt(l):
        p = lm[int(l)]
        return np.array([p.x, p.y], dtype=np.float32), float(p.visibility)

    kp = {}
    for side, suf in [("LEFT", "G"), ("RIGHT", "D")]:
        kp[f"Epaule {suf}"], kp[f"Epaule {suf} vis"] = pt(getattr(L, f"{side}_SHOULDER"))
        kp[f"Hanche {suf}"], kp[f"Hanche {suf} vis"] = pt(getattr(L, f"{side}_HIP"))
        kp[f"Genou {suf}"], kp[f"Genou {suf} vis"] = pt(getattr(L, f"{side}_KNEE"))
        kp[f"Cheville {suf}"], kp[f"Cheville {suf} vis"] = pt(getattr(L, f"{side}_ANKLE"))
        kp[f"Talon {suf}"], kp[f"Talon {suf} vis"] = pt(getattr(L, f"{side}_HEEL"))
        kp[f"Orteil {suf}"], kp[f"Orteil {suf} vis"] = pt(getattr(L, f"{side}_FOOT_INDEX"))
    return kp

# ==============================
# ANGLES FRONTAUX
# ==============================
def angle_3pts(a, b, c):
    """
    Angle géométrique (0-180°) au point b entre les segments b->a et b->c.
    """
    ba = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    bc = np.asarray(c, dtype=float) - np.asarray(b, dtype=float)

    # repère image -> repère math
    ba[1] *= -1
    bc[1] *= -1

    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-9
    cosv = np.dot(ba, bc) / denom
    cosv = np.clip(cosv, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosv)))

def signed_angle_vs_vertical(p1, p2):
    """
    Angle signé du segment p1->p2 par rapport à la verticale.
    """
    v = np.asarray(p2, dtype=float) - np.asarray(p1, dtype=float)
    vx = v[0]
    vy = -(v[1])
    return float(np.degrees(np.arctan2(vx, vy + 1e-9)))

def signed_angle_vs_horizontal(p1, p2):
    """
    Angle signé du segment p1->p2 par rapport à l'horizontale.
    """
    v = np.asarray(p2, dtype=float) - np.asarray(p1, dtype=float)
    vx = v[0]
    vy = -(v[1])
    return float(np.degrees(np.arctan2(vy, vx + 1e-9)))

def angle_genou_frontal(hip, knee, ankle, side=None):
    """
    Déviation frontale du genou = angle cuisse/jambe.
    0° ~ alignement neutre.
    """
    raw = angle_3pts(hip, knee, ankle)   # proche de 180 si aligné
    dev = 180.0 - raw                    # proche de 0 si aligné

    hip = np.asarray(hip, dtype=float)
    knee = np.asarray(knee, dtype=float)
    ankle = np.asarray(ankle, dtype=float)

    # Signe basé sur la position du genou par rapport à la ligne hanche-cheville
    v1 = knee - hip
    v2 = ankle - hip
    cross = v1[0] * v2[1] - v1[1] * v2[0]

    if side == "G":
        sign = -1.0 if cross > 0 else 1.0
    elif side == "D":
        sign = 1.0 if cross > 0 else -1.0
    else:
        sign = 1.0 if cross > 0 else -1.0

    return float(sign * abs(dev))

def angle_arriere_pied_frontal(ankle, heel, side=None):
    """
    Déviation frontale stable de l'arrière-pied par rapport à la verticale.
    On utilise le segment talon -> cheville, avec formule stable
    pour éviter les sauts absurdes de type atan2 sur quadrant.
    """
    ankle = np.asarray(ankle, dtype=float)
    heel = np.asarray(heel, dtype=float)

    # segment talon -> cheville
    v = ankle - heel
    dx = v[0]
    dy = v[1]

    # déviation par rapport à la verticale, version stable
    ang = np.degrees(np.arctan2(dx, abs(dy) + 1e-9))

    # harmonisation visuelle G/D
    if side == "G":
        ang = -ang

    return float(ang)

def angle_bassin_frontal(hipL, hipR):
    """
    Obliquité pelvienne.
    """
    return signed_angle_vs_horizontal(hipL, hipR)

def angle_tronc_frontal(shL, shR, hipL, hipR):
    """
    Inclinaison latérale du tronc.
    """
    mid_sh = (np.asarray(shL) + np.asarray(shR)) / 2.0
    mid_hip = (np.asarray(hipL) + np.asarray(hipR)) / 2.0
    return signed_angle_vs_vertical(mid_hip, mid_sh)

# ==============================
# CONTACTS SOL + CYCLE
# ==============================
def detect_foot_contacts(y, fps=FPS):
    y = np.asarray(y, dtype=float)

    if np.isnan(y).any():
        idx = np.arange(len(y))
        ok = ~np.isnan(y)
        if ok.sum() >= 2:
            y = np.interp(idx, idx[ok], y[ok])
        else:
            return np.array([], dtype=int), y

    y_s = smooth_clinical(y, smooth_level=2)

    inv = -y_s
    min_distance = max(1, int(0.35 * fps))
    prominence = max(1e-6, np.std(inv) * 0.2)

    peaks, _ = find_peaks(inv, distance=min_distance, prominence=prominence)
    return peaks, y_s

def compute_step_times(contact_idx, fps=FPS):
    contact_idx = np.asarray(contact_idx, dtype=int)
    if len(contact_idx) < 2:
        return [], None, None

    step_times = np.diff(contact_idx) / float(fps)
    return step_times.tolist(), float(np.mean(step_times)), float(np.std(step_times))

def detect_cycle(y):
    contacts, _ = detect_foot_contacts(y, fps=FPS)
    if len(contacts) < 2:
        return None

    mid = len(contacts) // 2
    if mid == 0:
        return int(contacts[0]), int(contacts[1])

    return int(contacts[mid - 1]), int(contacts[mid])

# ==============================
# VIDEO PROCESS
# ==============================
def process_video(path, conf):
    cap = cv2.VideoCapture(path)

    res = {
        "Genou G": [],
        "Genou D": [],
        "Arriere-pied G": [],
        "Arriere-pied D": [],
        "Bassin": [],
        "Tronc": [],
    }

    heelG_y, heelD_y = [], []
    frames = []

    while cap.isOpened():
        r, f = cap.read()
        if not r:
            break
        frames.append(f.copy())

        kp = detect_pose(f)
        if kp is None:
            for k in res:
                res[k].append(np.nan)
            heelG_y.append(np.nan)
            heelD_y.append(np.nan)
            continue

        def ok(n):
            return kp.get(f"{n} vis", 0.0) >= conf

        # Genou frontal = angle cuisse / jambe
        res["Genou G"].append(
            angle_genou_frontal(kp["Hanche G"], kp["Genou G"], kp["Cheville G"], side="G")
            if (ok("Hanche G") and ok("Genou G") and ok("Cheville G")) else np.nan
        )
        res["Genou D"].append(
            angle_genou_frontal(kp["Hanche D"], kp["Genou D"], kp["Cheville D"], side="D")
            if (ok("Hanche D") and ok("Genou D") and ok("Cheville D")) else np.nan
        )

        # Arrière-pied frontal stable
        res["Arriere-pied G"].append(
            angle_arriere_pied_frontal(kp["Cheville G"], kp["Talon G"], side="G")
            if (ok("Cheville G") and ok("Talon G")) else np.nan
        )
        res["Arriere-pied D"].append(
            angle_arriere_pied_frontal(kp["Cheville D"], kp["Talon D"], side="D")
            if (ok("Cheville D") and ok("Talon D")) else np.nan
        )

        # Bassin
        res["Bassin"].append(
            angle_bassin_frontal(kp["Hanche G"], kp["Hanche D"])
            if (ok("Hanche G") and ok("Hanche D")) else np.nan
        )

        # Tronc
        res["Tronc"].append(
            angle_tronc_frontal(kp["Epaule G"], kp["Epaule D"], kp["Hanche G"], kp["Hanche D"])
            if (ok("Epaule G") and ok("Epaule D") and ok("Hanche G") and ok("Hanche D")) else np.nan
        )

        heelG_y.append(float(kp["Talon G"][1]) if ok("Talon G") else np.nan)
        heelD_y.append(float(kp["Talon D"][1]) if ok("Talon D") else np.nan)

    cap.release()
    return res, heelG_y, heelD_y, frames

# ==============================
# ANNOTATION IMAGES
# ==============================
def draw_segment_with_angle(img_bgr, p1, p2, ang_deg, label, color=(0, 255, 0)):
    h, w = img_bgr.shape[:2]
    P1 = (int(p1[0] * w), int(p1[1] * h))
    P2 = (int(p2[0] * w), int(p2[1] * h))

    cv2.line(img_bgr, P1, P2, color, 4)
    cv2.circle(img_bgr, P1, 6, (0, 0, 255), -1)
    cv2.circle(img_bgr, P2, 6, (0, 0, 255), -1)

    txt = f"{label}: {ang_deg:.1f}°"
    tx, ty = P2[0] + 10, P2[1] - 10
    cv2.rectangle(img_bgr, (tx - 4, ty - 30), (tx + 190, ty + 6), (0, 0, 0), -1)
    cv2.putText(
        img_bgr, txt, (tx, ty),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA
    )

def annotate_frame(frame_bgr, kp, conf=0.30):
    if kp is None:
        return frame_bgr

    def ok(n):
        return kp.get(f"{n} vis", 0.0) >= conf

    out = frame_bgr.copy()

    # Genou G
    if ok("Hanche G") and ok("Genou G") and ok("Cheville G"):
        draw_segment_with_angle(
            out, kp["Hanche G"], kp["Genou G"],
            angle_genou_frontal(kp["Hanche G"], kp["Genou G"], kp["Cheville G"], side="G"),
            "Genou G"
        )
        h, w = out.shape[:2]
        K = (int(kp["Genou G"][0] * w), int(kp["Genou G"][1] * h))
        A = (int(kp["Cheville G"][0] * w), int(kp["Cheville G"][1] * h))
        cv2.line(out, K, A, (0, 255, 0), 4)

    # Genou D
    if ok("Hanche D") and ok("Genou D") and ok("Cheville D"):
        draw_segment_with_angle(
            out, kp["Hanche D"], kp["Genou D"],
            angle_genou_frontal(kp["Hanche D"], kp["Genou D"], kp["Cheville D"], side="D"),
            "Genou D"
        )
        h, w = out.shape[:2]
        K = (int(kp["Genou D"][0] * w), int(kp["Genou D"][1] * h))
        A = (int(kp["Cheville D"][0] * w), int(kp["Cheville D"][1] * h))
        cv2.line(out, K, A, (0, 255, 0), 4)

    # Arrière-pied G
    if ok("Cheville G") and ok("Talon G"):
        draw_segment_with_angle(
            out, kp["Talon G"], kp["Cheville G"],
            angle_arriere_pied_frontal(kp["Cheville G"], kp["Talon G"], side="G"),
            "AP G"
        )

    # Arrière-pied D
    if ok("Cheville D") and ok("Talon D"):
        draw_segment_with_angle(
            out, kp["Talon D"], kp["Cheville D"],
            angle_arriere_pied_frontal(kp["Cheville D"], kp["Talon D"], side="D"),
            "AP D"
        )

    # Bassin
    if ok("Hanche G") and ok("Hanche D"):
        draw_segment_with_angle(
            out, kp["Hanche G"], kp["Hanche D"],
            angle_bassin_frontal(kp["Hanche G"], kp["Hanche D"]),
            "Bassin"
        )

    # Tronc
    if ok("Epaule G") and ok("Epaule D") and ok("Hanche G") and ok("Hanche D"):
        mid_sh = (kp["Epaule G"] + kp["Epaule D"]) / 2.0
        mid_hip = (kp["Hanche G"] + kp["Hanche D"]) / 2.0
        draw_segment_with_angle(
            out, mid_hip, mid_sh,
            angle_tronc_frontal(kp["Epaule G"], kp["Epaule D"], kp["Hanche G"], kp["Hanche D"]),
            "Tronc"
        )

    return out

# ==============================
# ASYMETRIE
# ==============================
def asym_percent(left, right):
    if left is None or right is None:
        return None
    denom = (left + right) / 2.0
    if abs(denom) < 1e-6:
        return None
    return 100.0 * abs(right - left) / abs(denom)

# ==============================
# PDF EXPORT
# ==============================
def export_pdf(patient, keyframe_path, figures, table_data, annotated_images,
               asym_table=None, temporal_info=None, contact_fig_path=None):
    out_path = os.path.join(tempfile.gettempdir(), f"GaitScan_{patient['nom']}_{patient['prenom']}.pdf")

    doc = SimpleDocTemplate(
        out_path, pagesize=A4,
        leftMargin=1.7 * cm, rightMargin=1.7 * cm,
        topMargin=1.7 * cm, bottomMargin=1.7 * cm
    )

    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>GaitScan Pro – Analyse Frontale / Vue arrière</b>", styles["Title"]))
    story.append(Spacer(1, 0.2 * cm))

    story.append(Paragraph(
        f"<b>Patient :</b> {patient['nom']} {patient['prenom']}<br/>"
        f"<b>Date :</b> {datetime.now().strftime('%d/%m/%Y')}<br/>"
        f"<b>Angle de film :</b> {patient.get('camera','N/A')}<br/>"
        f"<b>Affichage phases :</b> {patient.get('phase','N/A')}<br/>"
        f"<b>Norme affichée :</b> {'Oui' if patient.get('show_norm', True) else 'Non'}<br/>"
        f"<b>Taille :</b> {patient.get('taille_cm','N/A')} cm",
        styles["Normal"]
    ))
    story.append(Spacer(1, 0.35 * cm))

    if temporal_info is not None:
        story.append(Paragraph("<b>Paramètres temporels</b>", styles["Heading2"]))
        txt = ""

        if temporal_info.get("G_mean") is not None:
            txt += (
                f"<b>Temps du pas Gauche :</b> {temporal_info['G_mean']:.2f} s "
                f"(± {temporal_info['G_std']:.2f} s)<br/>"
            )
        else:
            txt += "<b>Temps du pas Gauche :</b> non calculable<br/>"

        if temporal_info.get("D_mean") is not None:
            txt += (
                f"<b>Temps du pas Droit :</b> {temporal_info['D_mean']:.2f} s "
                f"(± {temporal_info['D_std']:.2f} s)<br/>"
            )
        else:
            txt += "<b>Temps du pas Droit :</b> non calculable<br/>"

        txt += (
            f"<b>Contacts détectés :</b> Gauche = {temporal_info.get('nG', 0)} "
            f"&nbsp;&nbsp; Droit = {temporal_info.get('nD', 0)}<br/>"
            "<i>Les contacts au sol sont estimés à partir des minima verticaux des talons.</i>"
        )

        story.append(Paragraph(txt, styles["Normal"]))
        story.append(Spacer(1, 0.25 * cm))

    if contact_fig_path is not None and os.path.exists(contact_fig_path):
        story.append(Paragraph("<b>Contacts au sol (talons)</b>", styles["Heading2"]))
        story.append(PDFImage(contact_fig_path, width=16 * cm, height=6 * cm))
        story.append(Spacer(1, 0.3 * cm))

    if asym_table:
        story.append(Paragraph("<b>Asymétries droite/gauche</b>", styles["Heading2"]))
        t = Table([["Mesure", "Moy G", "Moy D", "Asym %"]] + asym_table,
                  colWidths=[6 * cm, 3 * cm, 3 * cm, 3 * cm])
        t.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.7, colors.black),
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("ALIGN", (1, 1), (-1, -1), "CENTER")
        ]))
        story.append(t)
        story.append(Spacer(1, 0.35 * cm))

    story.append(Paragraph("<b>Image clé</b>", styles["Heading2"]))
    story.append(PDFImage(keyframe_path, width=16 * cm, height=8 * cm))
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("<b>Analyse frontale</b>", styles["Heading2"]))
    story.append(Spacer(1, 0.2 * cm))
    for joint, figpath in figures.items():
        story.append(Paragraph(f"<b>{joint}</b>", styles["Heading3"]))
        story.append(PDFImage(figpath, width=16 * cm, height=6 * cm))
        story.append(Spacer(1, 0.3 * cm))

    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph("<b>Synthèse (°)</b>", styles["Heading2"]))

    table = Table([["Mesure", "Min", "Moyenne", "Max"]] + table_data,
                  colWidths=[7 * cm, 3 * cm, 3 * cm, 3 * cm])
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.7, colors.black),
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("ALIGN", (1, 1), (-1, -1), "CENTER")
    ]))
    story.append(table)

    if annotated_images:
        story.append(PageBreak())
        story.append(Paragraph("<b>Images annotées</b>", styles["Heading2"]))
        story.append(Spacer(1, 0.2 * cm))
        for img in annotated_images:
            story.append(PDFImage(img, width=16 * cm, height=8 * cm))
            story.append(Spacer(1, 0.25 * cm))

    doc.build(story)
    return out_path

# ==============================
# PDF VIEW + PRINT
# ==============================
def pdf_viewer_with_print(pdf_bytes: bytes, height=800):
    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    html = f"""
    <div style="display:flex; gap:12px; align-items:center; margin: 6px 0 10px 0;">
      <button onclick="printPdf()" style="padding:10px 14px; font-size:16px; cursor:pointer;">
        🖨️ Imprimer le rapport
      </button>
      <span style="opacity:0.7;">(ouvre la boîte d’impression du navigateur)</span>
    </div>
    <iframe id="pdfFrame" src="data:application/pdf;base64,{b64}" width="100%" height="{height}px" style="border:1px solid #ddd; border-radius:8px;"></iframe>
    <script>
      function printPdf() {{
        const iframe = document.getElementById('pdfFrame');
        iframe.contentWindow.focus();
        iframe.contentWindow.print();
      }}
    </script>
    """
    components.html(html, height=height + 80, scrolling=True)

# ==============================
# UI
# ==============================
with st.sidebar:
    nom = st.text_input("Nom", "DURAND")
    prenom = st.text_input("Prénom", "Jean")
    camera_pos = st.selectbox("Angle de film", ["Devant", "Derrière"])
    phase_cote = st.selectbox("Phases", ["Aucune", "Droite", "Gauche", "Les deux"])
    smooth = st.slider("Lissage (patient)", 0, 10, 3)
    conf = st.slider("Seuil confiance", 0.1, 0.9, 0.3, 0.05)

    taille_cm = st.number_input("Taille du patient (cm)", min_value=80, max_value=230, value=170, step=1)

    show_norm = st.checkbox("Afficher la norme", value=True)
    norm_smooth_win = st.slider(
        "Lissage norme (simple)", 1, 21, 7, 2,
        help="Moyenne glissante. 1 = pas de lissage."
    )

video = st.file_uploader("Vidéo", ["mp4", "avi", "mov"])

# ==============================
# ANALYSE
# ==============================
if video and st.button("▶ Lancer l'analyse"):
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.write(video.read())
    tmp.close()

    data, heelG, heelD, frames = process_video(tmp.name, conf)
    os.unlink(tmp.name)

    contactsG, heelG_s = detect_foot_contacts(heelG, fps=FPS)
    contactsD, heelD_s = detect_foot_contacts(heelD, fps=FPS)

    step_times_G, step_time_G_mean, step_time_G_std = compute_step_times(contactsG, fps=FPS)
    step_times_D, step_time_D_mean, step_time_D_std = compute_step_times(contactsD, fps=FPS)

    phases = []
    if phase_cote in ["Gauche", "Les deux"]:
        c = detect_cycle(heelG)
        if c:
            phases.append((*c, "orange"))
    if phase_cote in ["Droite", "Les deux"]:
        c = detect_cycle(heelD)
        if c:
            phases.append((*c, "blue"))

    st.subheader("📐 Paramètres frontaux")
    st.caption("Analyse frontale : genou (angle cuisse-jambe), arrière-pied, bassin et tronc.")

    st.subheader("⏱️ Temps du pas")
    col1, col2 = st.columns(2)

    with col1:
        if step_time_G_mean is not None:
            st.write(f"**Temps du pas Gauche :** {step_time_G_mean:.2f} s")
            st.write(f"**Variabilité Gauche :** ± {step_time_G_std:.2f} s")
            st.write(f"**Nombre de contacts Gauche :** {len(contactsG)}")
        else:
            st.write("**Temps du pas Gauche :** non calculable")

    with col2:
        if step_time_D_mean is not None:
            st.write(f"**Temps du pas Droit :** {step_time_D_mean:.2f} s")
            st.write(f"**Variabilité Droit :** ± {step_time_D_std:.2f} s")
            st.write(f"**Nombre de contacts Droit :** {len(contactsD)}")
        else:
            st.write("**Temps du pas Droit :** non calculable")

    st.caption("Les contacts au sol sont estimés à partir des minima verticaux des talons.")

    keyframe_path = os.path.join(tempfile.gettempdir(), "keyframe.png")
    cv2.imwrite(keyframe_path, frames[len(frames) // 2])

    figures = {}
    table_data = []
    asym_rows = []

    metrics_pairs = [
        ("Genou", "Genou G", "Genou D"),
        ("Arriere-pied", "Arriere-pied G", "Arriere-pied D"),
    ]

    metrics_single = [
        ("Bassin", "Bassin"),
        ("Tronc", "Tronc"),
    ]

    # Courbes bilatérales
    for label, left_key, right_key in metrics_pairs:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [2, 1]})

        g_raw = np.array(data[left_key], dtype=float)
        d_raw = np.array(data[right_key], dtype=float)

        local_smooth = smooth + 2 if label == "Arriere-pied" else smooth
        g = smooth_clinical(g_raw, smooth_level=local_smooth)
        d = smooth_clinical(d_raw, smooth_level=local_smooth)

        ax1.plot(g, label="Gauche", color="red")
        ax1.plot(d, label="Droite", color="blue")
        ax1.axhline(0, linestyle="--", linewidth=1, color="black", alpha=0.6)
        ax1.set_ylim(-25, 25)
        for c0, c1, col in phases:
            ax1.axvspan(c0, c1, color=col, alpha=0.3)
        ax1.set_title(f"{label} – Analyse frontale")
        ax1.set_ylabel("Angle (°)")
        ax1.legend()

        if show_norm:
            norm = norm_curve(left_key, len(g))
            norm = smooth_ma(norm, win=norm_smooth_win)
            ax2.plot(norm, color="green")
            ax2.axhline(0, linestyle="--", linewidth=1, color="black", alpha=0.6)
            ax2.set_ylim(-25, 25)
            ax2.set_title("Norme (indicative)")
        else:
            ax2.axis("off")

        st.pyplot(fig)

        fig_path = os.path.join(tempfile.gettempdir(), f"{label}_plot.png")
        fig.savefig(fig_path, bbox_inches="tight")
        plt.close(fig)
        figures[label] = fig_path

        def stats(arr_filtered, arr_raw):
            mask = ~np.isnan(arr_raw)
            if mask.sum() == 0:
                return np.nan, np.nan, np.nan, None
            vals = arr_filtered[mask]
            return float(np.min(vals)), float(np.mean(vals)), float(np.max(vals)), float(np.mean(vals))

        gmin, gmean, gmax, gmean_only = stats(g, g_raw)
        dmin, dmean, dmax, dmean_only = stats(d, d_raw)

        table_data.append([f"{label} Gauche", f"{gmin:.1f}", f"{gmean:.1f}", f"{gmax:.1f}"])
        table_data.append([f"{label} Droite", f"{dmin:.1f}", f"{dmean:.1f}", f"{dmax:.1f}"])

        a = asym_percent(gmean_only, dmean_only)
        if a is None:
            asym_rows.append([
                label,
                f"{gmean_only:.1f}" if gmean_only is not None else "NA",
                f"{dmean_only:.1f}" if dmean_only is not None else "NA",
                "NA"
            ])
        else:
            asym_rows.append([label, f"{gmean_only:.1f}", f"{dmean_only:.1f}", f"{a:.1f}"])

    # Courbes globales
    for label, key in metrics_single:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [2, 1]})

        raw = np.array(data[key], dtype=float)
        val = smooth_clinical(raw, smooth_level=smooth)

        ax1.plot(val, label=label, color="purple")
        ax1.axhline(0, linestyle="--", linewidth=1, color="black", alpha=0.6)
        ax1.set_ylim(-15, 15)
        for c0, c1, col in phases:
            ax1.axvspan(c0, c1, color=col, alpha=0.3)
        ax1.set_title(f"{label} – Analyse frontale")
        ax1.set_ylabel("Angle (°)")
        ax1.legend()

        if show_norm:
            norm = norm_curve(key, len(val))
            norm = smooth_ma(norm, win=norm_smooth_win)
            ax2.plot(norm, color="green")
            ax2.axhline(0, linestyle="--", linewidth=1, color="black", alpha=0.6)
            ax2.set_ylim(-15, 15)
            ax2.set_title("Norme (indicative)")
        else:
            ax2.axis("off")

        st.pyplot(fig)

        fig_path = os.path.join(tempfile.gettempdir(), f"{label}_plot.png")
        fig.savefig(fig_path, bbox_inches="tight")
        plt.close(fig)
        figures[label] = fig_path

        mask = ~np.isnan(raw)
        if mask.sum() == 0:
            vmin = vmean = vmax = np.nan
        else:
            vals = val[mask]
            vmin = float(np.min(vals))
            vmean = float(np.mean(vals))
            vmax = float(np.max(vals))

        table_data.append([label, f"{vmin:.1f}", f"{vmean:.1f}", f"{vmax:.1f}"])

    st.subheader("↔️ Asymétries droite/gauche")
    for row in asym_rows:
        st.write(f"**{row[0]}** — Moy G: {row[1]}° | Moy D: {row[2]}° | Asym: {row[3]}%")

    st.subheader("🦶 Contacts au sol (talons)")
    fig_contact, ax = plt.subplots(figsize=(12, 4))

    x = np.arange(len(heelG_s)) / FPS
    ax.plot(x, heelG_s, label="Talon Gauche", color="red")
    ax.plot(x, heelD_s, label="Talon Droit", color="blue")

    if len(contactsG) > 0:
        ax.plot(contactsG / FPS, heelG_s[contactsG], "o", color="red")
    for c in contactsG:
        ax.axvline(c / FPS, color="red", alpha=0.15)

    if len(contactsD) > 0:
        ax.plot(contactsD / FPS, heelD_s[contactsD], "o", color="blue")
    for c in contactsD:
        ax.axvline(c / FPS, color="blue", alpha=0.15)

    ax.set_title("Détection des contacts au sol")
    ax.set_xlabel("Temps (s)")
    ax.legend()
    st.pyplot(fig_contact)

    contact_fig_path = os.path.join(tempfile.gettempdir(), "contacts_sol.png")
    fig_contact.savefig(contact_fig_path, bbox_inches="tight")
    plt.close(fig_contact)

    st.subheader("📸 Captures annotées")
    num_photos = st.slider("Nombre d'images extraites", 1, 10, 3)
    total_frames = len(frames)
    idxs = np.linspace(0, total_frames - 1, num_photos, dtype=int)

    annotated_images = []
    for i, idx in enumerate(idxs):
        frame = frames[idx]
        kp = detect_pose(frame)
        ann = annotate_frame(frame, kp, conf=conf)

        out_img = os.path.join(tempfile.gettempdir(), f"annotated_{i}.png")
        cv2.imwrite(out_img, ann)
        annotated_images.append(out_img)

        st.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB), caption=f"Image annotée {i+1} (frame {idx})")

    temporal_info = {
        "G_mean": step_time_G_mean,
        "G_std": step_time_G_std,
        "D_mean": step_time_D_mean,
        "D_std": step_time_D_std,
        "nG": len(contactsG),
        "nD": len(contactsD),
    }

    pdf_path = export_pdf(
        patient={
            "nom": nom,
            "prenom": prenom,
            "camera": camera_pos,
            "phase": phase_cote,
            "taille_cm": int(taille_cm),
            "show_norm": bool(show_norm)
        },
        keyframe_path=keyframe_path,
        figures=figures,
        table_data=table_data,
        annotated_images=annotated_images,
        asym_table=asym_rows,
        temporal_info=temporal_info,
        contact_fig_path=contact_fig_path
    )

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    st.success("✅ Rapport généré")
    st.download_button(
        "📄 Télécharger le rapport PDF",
        data=pdf_bytes,
        file_name=f"GaitScan_{nom}_{prenom}.pdf",
        mime="application/pdf"
    )
"""

CINEMATIQUE_CODE = r"""
import streamlit as st
import cv2, os, tempfile, base64
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.signal import find_peaks
import mediapipe as mp

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Image as PDFImage,
    Spacer, Table, TableStyle, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors

import streamlit.components.v1 as components

# ==============================
# CONFIG
# ==============================
st.set_page_config("GaitScan Pro (MediaPipe)", layout="wide")
st.title("🏃 GaitScan Pro – Analyse Cinématique")
FPS = 30

# ==============================
# MEDIAPIPE
# ==============================
mp_pose = mp.solutions.pose

@st.cache_resource
def load_pose():
    return mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

pose = load_pose()

# ==============================
# NORMES
# ==============================
def norm_curve(joint, n):
    x = np.linspace(0, 100, n)
    if joint == "Genou":
        return np.interp(x, [0, 15, 40, 60, 80, 100], [5, 15, 5, 40, 60, 5])
    if joint == "Hanche":
        return np.interp(x, [0, 30, 60, 100], [30, 0, -10, 30])
    if joint == "Cheville":
        return np.interp(x, [0, 10, 50, 70, 100], [5, 10, 25, 10, 5])
    if joint == "Tronc":
        return np.zeros(n)
    return np.zeros(n)

def smooth_ma(y, win=7):
    y = np.asarray(y, dtype=float)
    if win is None or win <= 1:
        return y
    win = int(win)
    if win % 2 == 0:
        win += 1
    pad = win // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(win, dtype=float) / win
    return np.convolve(ypad, kernel, mode="valid")

# ==============================
# OUTLIERS + LISSAGE CLINIQUE
# ==============================
def interp_nan(arr):
    arr = np.asarray(arr, dtype=float)
    idx = np.arange(len(arr))
    ok = ~np.isnan(arr)
    if ok.sum() >= 2:
        return np.interp(idx, idx[ok], arr[ok])
    return np.zeros_like(arr)

def remove_outliers_hampel(x, win=5, n_sigmas=3.0):
    x = np.asarray(x, dtype=float).copy()
    n = len(x)
    if n < 3:
        return x

    y = x.copy()
    k = 1.4826

    for i in range(n):
        i0 = max(0, i - win)
        i1 = min(n, i + win + 1)
        w = x[i0:i1]
        med = np.median(w)
        mad = np.median(np.abs(w - med))

        if mad < 1e-9:
            continue

        if abs(x[i] - med) > n_sigmas * k * mad:
            y[i] = med

    return y

def smooth_clinical(arr, smooth_level=3):
    x = interp_nan(arr)
    x = remove_outliers_hampel(x, win=3 + smooth_level, n_sigmas=3.0)
    win = max(3, 2 * smooth_level + 3)
    if win % 2 == 0:
        win += 1
    return smooth_ma(x, win=win)

# ==============================
# POSE DETECTION
# ==============================
def detect_pose(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(img_rgb)
    if not res.pose_landmarks:
        return None

    lm = res.pose_landmarks.landmark
    L = mp_pose.PoseLandmark

    def pt(l):
        p = lm[int(l)]
        return np.array([p.x, p.y], dtype=np.float32), float(p.visibility)

    kp = {}
    for side, suf in [("LEFT", "G"), ("RIGHT", "D")]:
        kp[f"Epaule {suf}"], kp[f"Epaule {suf} vis"] = pt(getattr(L, f"{side}_SHOULDER"))
        kp[f"Hanche {suf}"], kp[f"Hanche {suf} vis"] = pt(getattr(L, f"{side}_HIP"))
        kp[f"Genou {suf}"], kp[f"Genou {suf} vis"] = pt(getattr(L, f"{side}_KNEE"))
        kp[f"Cheville {suf}"], kp[f"Cheville {suf} vis"] = pt(getattr(L, f"{side}_ANKLE"))
        kp[f"Talon {suf}"], kp[f"Talon {suf} vis"] = pt(getattr(L, f"{side}_HEEL"))
        kp[f"Orteil {suf}"], kp[f"Orteil {suf} vis"] = pt(getattr(L, f"{side}_FOOT_INDEX"))
    return kp

# ==============================
# ANGLES
# ==============================
def angle(a, b, c):
    ba = a - b
    bc = c - b
    ba[1] *= -1
    bc[1] *= -1
    cosv = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cosv, -1, 1))))

def angle_between(v1, v2):
    v1 = np.asarray(v1, dtype=float).copy()
    v2 = np.asarray(v2, dtype=float).copy()
    v1[1] *= -1
    v2[1] *= -1
    cosv = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cosv, -1, 1))))

def angle_hanche(e, h, g):
    return 180 - angle(e, h, g)

def angle_genou(h, g, c):
    return 180 - angle(h, g, c)

def angle_cheville_brut(g, c, t, o):
    jambe = g - c
    pied = o - t
    return angle_between(jambe, pied)

def angle_cheville(g, c, t, o):
    return angle_cheville_brut(g, c, t, o) - 90.0

def midpoint(p1, p2):
    return (np.asarray(p1, dtype=float) + np.asarray(p2, dtype=float)) / 2.0

def angle_tronc(epaule_g, epaule_d, hanche_g, hanche_d):
    ep_mid = midpoint(epaule_g, epaule_d)
    ha_mid = midpoint(hanche_g, hanche_d)
    v = ep_mid - ha_mid
    v[1] *= -1
    return float(np.degrees(np.arctan2(v[0], v[1] + 1e-6)))

# ==============================
# CONTACTS SOL + CYCLE
# ==============================
def detect_foot_contacts(y, fps=FPS):
    y = np.asarray(y, dtype=float)

    if np.isnan(y).any():
        idx = np.arange(len(y))
        ok = ~np.isnan(y)
        if ok.sum() >= 2:
            y = np.interp(idx, idx[ok], y[ok])
        else:
            return np.array([], dtype=int), y

    y_s = smooth_clinical(y, smooth_level=2)

    inv = -y_s
    min_distance = max(1, int(0.35 * fps))
    prominence = max(1e-6, np.std(inv) * 0.2)

    peaks, _ = find_peaks(inv, distance=min_distance, prominence=prominence)
    return peaks, y_s

def compute_step_times(contact_idx, fps=FPS):
    contact_idx = np.asarray(contact_idx, dtype=int)
    if len(contact_idx) < 2:
        return [], None, None

    step_times = np.diff(contact_idx) / float(fps)
    return step_times.tolist(), float(np.mean(step_times)), float(np.std(step_times))

def detect_cycle(y):
    contacts, _ = detect_foot_contacts(y, fps=FPS)
    if len(contacts) < 2:
        return None

    mid = len(contacts) // 2
    if mid == 0:
        return int(contacts[0]), int(contacts[1])

    return int(contacts[mid - 1]), int(contacts[mid])

# ==============================
# VIDEO PROCESS
# ==============================
def process_video(path, conf):
    cap = cv2.VideoCapture(path)
    res = {k: [] for k in ["Tronc", "Hanche G", "Hanche D", "Genou G", "Genou D", "Cheville G", "Cheville D"]}

    heelG_y, heelD_y = [], []
    heelG_x, heelD_x = [], []
    toeG_x, toeD_x = [], []

    frames = []

    while cap.isOpened():
        r, f = cap.read()
        if not r:
            break
        frames.append(f.copy())

        kp = detect_pose(f)
        if kp is None:
            for k in res:
                res[k].append(np.nan)
            heelG_y.append(np.nan)
            heelD_y.append(np.nan)
            heelG_x.append(np.nan)
            heelD_x.append(np.nan)
            toeG_x.append(np.nan)
            toeD_x.append(np.nan)
            continue

        def ok(n):
            return kp.get(f"{n} vis", 0.0) >= conf

        res["Tronc"].append(
            angle_tronc(kp["Epaule G"], kp["Epaule D"], kp["Hanche G"], kp["Hanche D"])
            if (ok("Epaule G") and ok("Epaule D") and ok("Hanche G") and ok("Hanche D")) else np.nan
        )

        res["Hanche G"].append(
            angle_hanche(kp["Epaule G"], kp["Hanche G"], kp["Genou G"])
            if (ok("Epaule G") and ok("Hanche G") and ok("Genou G")) else np.nan
        )
        res["Hanche D"].append(
            angle_hanche(kp["Epaule D"], kp["Hanche D"], kp["Genou D"])
            if (ok("Epaule D") and ok("Hanche D") and ok("Genou D")) else np.nan
        )

        res["Genou G"].append(
            angle_genou(kp["Hanche G"], kp["Genou G"], kp["Cheville G"])
            if (ok("Hanche G") and ok("Genou G") and ok("Cheville G")) else np.nan
        )
        res["Genou D"].append(
            angle_genou(kp["Hanche D"], kp["Genou D"], kp["Cheville D"])
            if (ok("Hanche D") and ok("Genou D") and ok("Cheville D")) else np.nan
        )

        res["Cheville G"].append(
            angle_cheville(kp["Genou G"], kp["Cheville G"], kp["Talon G"], kp["Orteil G"])
            if (ok("Genou G") and ok("Cheville G") and ok("Talon G") and ok("Orteil G")) else np.nan
        )
        res["Cheville D"].append(
            angle_cheville(kp["Genou D"], kp["Cheville D"], kp["Talon D"], kp["Orteil D"])
            if (ok("Genou D") and ok("Cheville D") and ok("Talon D") and ok("Orteil D")) else np.nan
        )

        heelG_y.append(float(kp["Talon G"][1]) if ok("Talon G") else np.nan)
        heelD_y.append(float(kp["Talon D"][1]) if ok("Talon D") else np.nan)

        heelG_x.append(float(kp["Talon G"][0]) if ok("Talon G") else np.nan)
        heelD_x.append(float(kp["Talon D"][0]) if ok("Talon D") else np.nan)

        toeG_x.append(float(kp["Orteil G"][0]) if ok("Orteil G") else np.nan)
        toeD_x.append(float(kp["Orteil D"][0]) if ok("Orteil D") else np.nan)

    cap.release()
    return res, heelG_y, heelD_y, heelG_x, heelD_x, toeG_x, toeD_x, frames

# ==============================
# ANNOTATION IMAGES
# ==============================
def draw_angle_on_frame(img_bgr, pA, pB, pC, ang_deg, color=(0, 255, 0)):
    h, w = img_bgr.shape[:2]
    A = (int(pA[0] * w), int(pA[1] * h))
    B = (int(pB[0] * w), int(pB[1] * h))
    C = (int(pC[0] * w), int(pC[1] * h))

    line_th = 4
    circle_r = 7
    text_scale = 1.2
    text_th = 3

    cv2.line(img_bgr, A, B, color, line_th)
    cv2.line(img_bgr, C, B, color, line_th)
    cv2.circle(img_bgr, B, circle_r, (0, 0, 255), -1)

    label = f"{int(round(ang_deg))} deg"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_th)
    tx, ty = B[0] + 10, B[1] - 10
    cv2.rectangle(img_bgr, (tx - 4, ty - th - 6), (tx + tw + 6, ty + 6), (0, 0, 0), -1)
    cv2.putText(img_bgr, label, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 255, 255), text_th, cv2.LINE_AA)

def draw_ankle_angle_on_frame(img_bgr, knee, ankle, heel, toe, ang_deg, color=(0, 255, 0)):
    h, w = img_bgr.shape[:2]

    K = (int(knee[0] * w), int(knee[1] * h))
    A = (int(ankle[0] * w), int(ankle[1] * h))
    H = (int(heel[0] * w), int(heel[1] * h))
    T = (int(toe[0] * w), int(toe[1] * h))

    line_th = 4
    circle_r = 7
    text_scale = 1.2
    text_th = 3

    cv2.line(img_bgr, K, A, color, line_th)
    cv2.line(img_bgr, H, T, color, line_th)
    cv2.circle(img_bgr, A, circle_r, (0, 0, 255), -1)

    label = f"{int(round(ang_deg))} deg"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_th)
    tx, ty = A[0] + 10, A[1] - 10
    cv2.rectangle(img_bgr, (tx - 4, ty - th - 6), (tx + tw + 6, ty + 6), (0, 0, 0), -1)
    cv2.putText(img_bgr, label, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 255, 255), text_th, cv2.LINE_AA)

def draw_trunk_angle_on_frame(img_bgr, shoulder_mid, hip_mid, ang_deg, color=(255, 165, 0)):
    h, w = img_bgr.shape[:2]

    S = (int(shoulder_mid[0] * w), int(shoulder_mid[1] * h))
    H = (int(hip_mid[0] * w), int(hip_mid[1] * h))

    ref_len = int(0.18 * h)
    V = (H[0], H[1] - ref_len)

    line_th = 4
    circle_r = 7
    text_scale = 1.0
    text_th = 3

    cv2.line(img_bgr, H, S, color, line_th)
    cv2.line(img_bgr, H, V, (200, 200, 200), 2)
    cv2.circle(img_bgr, H, circle_r, (0, 0, 255), -1)

    label = f"Tronc {ang_deg:+.1f} deg"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_th)
    tx, ty = H[0] + 10, H[1] + 30
    cv2.rectangle(img_bgr, (tx - 4, ty - th - 6), (tx + tw + 6, ty + 6), (0, 0, 0), -1)
    cv2.putText(img_bgr, label, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 255, 255), text_th, cv2.LINE_AA)

def annotate_frame(frame_bgr, kp, conf=0.30):
    if kp is None:
        return frame_bgr

    def ok(n):
        return kp.get(f"{n} vis", 0.0) >= conf

    out = frame_bgr.copy()

    if ok("Epaule G") and ok("Epaule D") and ok("Hanche G") and ok("Hanche D"):
        shoulder_mid = midpoint(kp["Epaule G"], kp["Epaule D"])
        hip_mid = midpoint(kp["Hanche G"], kp["Hanche D"])
        draw_trunk_angle_on_frame(
            out,
            shoulder_mid,
            hip_mid,
            angle_tronc(kp["Epaule G"], kp["Epaule D"], kp["Hanche G"], kp["Hanche D"])
        )

    if ok("Epaule G") and ok("Hanche G") and ok("Genou G"):
        draw_angle_on_frame(out, kp["Epaule G"], kp["Hanche G"], kp["Genou G"],
                            angle_hanche(kp["Epaule G"], kp["Hanche G"], kp["Genou G"]))
    if ok("Epaule D") and ok("Hanche D") and ok("Genou D"):
        draw_angle_on_frame(out, kp["Epaule D"], kp["Hanche D"], kp["Genou D"],
                            angle_hanche(kp["Epaule D"], kp["Hanche D"], kp["Genou D"]))

    if ok("Hanche G") and ok("Genou G") and ok("Cheville G"):
        draw_angle_on_frame(out, kp["Hanche G"], kp["Genou G"], kp["Cheville G"],
                            angle_genou(kp["Hanche G"], kp["Genou G"], kp["Cheville G"]))
    if ok("Hanche D") and ok("Genou D") and ok("Cheville D"):
        draw_angle_on_frame(out, kp["Hanche D"], kp["Genou D"], kp["Cheville D"],
                            angle_genou(kp["Hanche D"], kp["Genou D"], kp["Cheville D"]))

    if ok("Genou G") and ok("Cheville G") and ok("Talon G") and ok("Orteil G"):
        draw_ankle_angle_on_frame(
            out,
            kp["Genou G"], kp["Cheville G"], kp["Talon G"], kp["Orteil G"],
            angle_cheville_brut(kp["Genou G"], kp["Cheville G"], kp["Talon G"], kp["Orteil G"])
        )
    if ok("Genou D") and ok("Cheville D") and ok("Talon D") and ok("Orteil D"):
        draw_ankle_angle_on_frame(
            out,
            kp["Genou D"], kp["Cheville D"], kp["Talon D"], kp["Orteil D"],
            angle_cheville_brut(kp["Genou D"], kp["Cheville D"], kp["Talon D"], kp["Orteil D"])
        )

    return out

# ==============================
# STEP LENGTH + ASYMMETRY
# ==============================
def nan_interp(x):
    x = np.array(x, dtype=float)
    idx = np.arange(len(x))
    ok = ~np.isnan(x)
    if ok.sum() >= 2:
        return np.interp(idx, idx[ok], x[ok])
    return None

def asym_percent(left, right):
    if left is None or right is None:
        return None
    denom = (left + right) / 2.0
    if abs(denom) < 1e-6:
        return None
    return 100.0 * abs(right - left) / abs(denom)

def compute_step_length_cm(heelG_y, heelD_y, heelG_x, heelD_x, toeG_x, toeD_x, taille_cm):
    """
    Longueur du pas estimée en 2D :
    - pas gauche = distance horizontale entre talon gauche à l'attaque
      et avant-pied droit
    - pas droit = distance horizontale entre talon droit à l'attaque
      et avant-pied gauche
    """

    contactsG, _ = detect_foot_contacts(heelG_y, fps=FPS)
    contactsD, _ = detect_foot_contacts(heelD_y, fps=FPS)

    hGx = nan_interp(heelG_x)
    hDx = nan_interp(heelD_x)
    tGx = nan_interp(toeG_x)
    tDx = nan_interp(toeD_x)

    if hGx is None or hDx is None or tGx is None or tDx is None:
        return None, None, None, None, None

    stepG_list = []
    stepD_list = []

    for i in contactsG:
        if 0 <= i < len(hGx) and 0 <= i < len(tDx):
            stepG_list.append(abs(hGx[i] - tDx[i]))

    for i in contactsD:
        if 0 <= i < len(hDx) and 0 <= i < len(tGx):
            stepD_list.append(abs(hDx[i] - tGx[i]))

    valid_norm = stepG_list + stepD_list
    if len(valid_norm) == 0:
        return None, None, None, None, None

    scale = float(taille_cm) / 0.53

    stepG_cm = float(np.mean(stepG_list) * scale) if len(stepG_list) > 0 else None
    stepD_cm = float(np.mean(stepD_list) * scale) if len(stepD_list) > 0 else None

    valid_cm = [v for v in [stepG_cm, stepD_cm] if v is not None]
    step_mean_cm = float(np.mean(valid_cm))
    step_std_cm = float(np.std(valid_cm))
    step_asym = asym_percent(stepG_cm, stepD_cm)

    return step_mean_cm, step_std_cm, stepG_cm, stepD_cm, step_asym

# ==============================
# PDF EXPORT
# ==============================
def export_pdf(patient, keyframe_path, figures, table_data, annotated_images,
               step_info=None, asym_table=None, temporal_info=None, contact_fig_path=None):
    out_path = os.path.join(tempfile.gettempdir(), f"GaitScan_{patient['nom']}_{patient['prenom']}.pdf")

    doc = SimpleDocTemplate(
        out_path, pagesize=A4,
        leftMargin=1.7 * cm, rightMargin=1.7 * cm,
        topMargin=1.7 * cm, bottomMargin=1.7 * cm
    )

    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>GaitScan Pro – Analyse Cinématique</b>", styles["Title"]))
    story.append(Spacer(1, 0.2 * cm))

    story.append(Paragraph(
        f"<b>Patient :</b> {patient['nom']} {patient['prenom']}<br/>"
        f"<b>Date :</b> {datetime.now().strftime('%d/%m/%Y')}<br/>"
        f"<b>Angle de film :</b> {patient.get('camera','N/A')}<br/>"
        f"<b>Affichage phases :</b> {patient.get('phase','N/A')}<br/>"
        f"<b>Norme affichée :</b> {'Oui' if patient.get('show_norm', True) else 'Non'}<br/>"
        f"<b>Taille :</b> {patient.get('taille_cm','N/A')} cm",
        styles["Normal"]
    ))
    story.append(Spacer(1, 0.35 * cm))

    if step_info is not None:
        story.append(Paragraph("<b>Paramètres spatio-temporels (estimation)</b>", styles["Heading2"]))
        story.append(Paragraph(
            f"<b>Longueur de pas moyenne :</b> {step_info['mean']:.1f} cm<br/>"
            f"<b>Variabilité :</b> ± {step_info['std']:.1f} cm<br/>"
            + (f"<b>Pas G :</b> {step_info['G']:.1f} cm &nbsp;&nbsp; <b>Pas D :</b> {step_info['D']:.1f} cm<br/>"
               if step_info.get("G") is not None and step_info.get("D") is not None else "")
            + (f"<b>Asymétrie pas (G/D) :</b> {step_info['asym']:.1f} %<br/>"
               if step_info.get("asym") is not None else "")
            + "<i>Mesure monocaméra 2D sans calibration métrique : valeurs estimées.</i>",
            styles["Normal"]
        ))
        story.append(Spacer(1, 0.25 * cm))

    if temporal_info is not None:
        story.append(Paragraph("<b>Paramètres temporels</b>", styles["Heading2"]))
        txt = ""

        if temporal_info.get("G_mean") is not None:
            txt += (
                f"<b>Temps du pas Gauche :</b> {temporal_info['G_mean']:.2f} s "
                f"(± {temporal_info['G_std']:.2f} s)<br/>"
            )
        else:
            txt += "<b>Temps du pas Gauche :</b> non calculable<br/>"

        if temporal_info.get("D_mean") is not None:
            txt += (
                f"<b>Temps du pas Droit :</b> {temporal_info['D_mean']:.2f} s "
                f"(± {temporal_info['D_std']:.2f} s)<br/>"
            )
        else:
            txt += "<b>Temps du pas Droit :</b> non calculable<br/>"

        txt += (
            f"<b>Contacts détectés :</b> Gauche = {temporal_info.get('nG', 0)} "
            f"&nbsp;&nbsp; Droit = {temporal_info.get('nD', 0)}<br/>"
            "<i>Les contacts au sol sont estimés à partir des minima verticaux des talons.</i>"
        )

        story.append(Paragraph(txt, styles["Normal"]))
        story.append(Spacer(1, 0.25 * cm))

    if contact_fig_path is not None and os.path.exists(contact_fig_path):
        story.append(Paragraph("<b>Contacts au sol (talons)</b>", styles["Heading2"]))
        story.append(PDFImage(contact_fig_path, width=16 * cm, height=6 * cm))
        story.append(Spacer(1, 0.3 * cm))

    if asym_table:
        story.append(Paragraph("<b>Asymétries droite/gauche (angles)</b>", styles["Heading2"]))
        t = Table([["Mesure", "Moy G", "Moy D", "Asym %"]] + asym_table,
                  colWidths=[6 * cm, 3 * cm, 3 * cm, 3 * cm])
        t.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.7, colors.black),
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("ALIGN", (1, 1), (-1, -1), "CENTER")
        ]))
        story.append(t)
        story.append(Spacer(1, 0.35 * cm))

    story.append(Paragraph("<b>Image clé</b>", styles["Heading2"]))
    story.append(PDFImage(keyframe_path, width=16 * cm, height=8 * cm))
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("<b>Analyse articulaire</b>", styles["Heading2"]))
    story.append(Spacer(1, 0.2 * cm))
    for joint, figpath in figures.items():
        story.append(Paragraph(f"<b>{joint}</b>", styles["Heading3"]))
        story.append(PDFImage(figpath, width=16 * cm, height=6 * cm))
        story.append(Spacer(1, 0.3 * cm))

    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph("<b>Synthèse (°)</b>", styles["Heading2"]))

    table = Table([["Mesure", "Min", "Moyenne", "Max"]] + table_data,
                  colWidths=[7 * cm, 3 * cm, 3 * cm, 3 * cm])
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.7, colors.black),
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("ALIGN", (1, 1), (-1, -1), "CENTER")
    ]))
    story.append(table)

    if annotated_images:
        story.append(PageBreak())
        story.append(Paragraph("<b>Images annotées (angles)</b>", styles["Heading2"]))
        story.append(Spacer(1, 0.2 * cm))
        for img in annotated_images:
            story.append(PDFImage(img, width=16 * cm, height=8 * cm))
            story.append(Spacer(1, 0.25 * cm))

    doc.build(story)
    return out_path

# ==============================
# PDF VIEW + PRINT (browser-side)
# ==============================
def pdf_viewer_with_print(pdf_bytes: bytes, height=800):
    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    html = f"""
    <div style="display:flex; gap:12px; align-items:center; margin: 6px 0 10px 0;">
      <button onclick="printPdf()" style="padding:10px 14px; font-size:16px; cursor:pointer;">
        🖨️ Imprimer le rapport
      </button>
      <span style="opacity:0.7;">(ouvre la boîte d’impression du navigateur)</span>
    </div>
    <iframe id="pdfFrame" src="data:application/pdf;base64,{b64}" width="100%" height="{height}px" style="border:1px solid #ddd; border-radius:8px;"></iframe>
    <script>
      function printPdf() {{
        const iframe = document.getElementById('pdfFrame');
        iframe.contentWindow.focus();
        iframe.contentWindow.print();
      }}
    </script>
    """
    components.html(html, height=height + 80, scrolling=True)

# ==============================
# UI
# ==============================
with st.sidebar:
    nom = st.text_input("Nom", "DURAND")
    prenom = st.text_input("Prénom", "Jean")
    camera_pos = st.selectbox("Angle de film", ["Devant", "Droite", "Gauche"])
    phase_cote = st.selectbox("Phases", ["Aucune", "Droite", "Gauche", "Les deux"])
    smooth = st.slider("Lissage (patient)", 0, 10, 3)
    conf = st.slider("Seuil confiance", 0.1, 0.9, 0.3, 0.05)

    taille_cm = st.number_input("Taille du patient (cm)", min_value=80, max_value=230, value=170, step=1)

    show_norm = st.checkbox("Afficher la norme", value=True)
    norm_smooth_win = st.slider(
        "Lissage norme (simple)", 1, 21, 7, 2,
        help="Moyenne glissante (impair conseillé). 1 = pas de lissage."
    )

video = st.file_uploader("Vidéo", ["mp4", "avi", "mov"])

# ==============================
# ANALYSE
# ==============================
if video and st.button("▶ Lancer l'analyse"):
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.write(video.read())
    tmp.close()

    data, heelG, heelD, heelG_x, heelD_x, toeG_x, toeD_x, frames = process_video(tmp.name, conf)
    os.unlink(tmp.name)

    contactsG, heelG_s = detect_foot_contacts(heelG, fps=FPS)
    contactsD, heelD_s = detect_foot_contacts(heelD, fps=FPS)

    step_times_G, step_time_G_mean, step_time_G_std = compute_step_times(contactsG, fps=FPS)
    step_times_D, step_time_D_mean, step_time_D_std = compute_step_times(contactsD, fps=FPS)

    phases = []
    if phase_cote in ["Gauche", "Les deux"]:
        c = detect_cycle(heelG)
        if c:
            phases.append((*c, "orange"))
    if phase_cote in ["Droite", "Les deux"]:
        c = detect_cycle(heelD)
        if c:
            phases.append((*c, "blue"))

    step_mean, step_std, stepG_cm, stepD_cm, step_asym = compute_step_length_cm(
        heelG, heelD, heelG_x, heelD_x, toeG_x, toeD_x, float(taille_cm)
    )

    st.subheader("📏 Paramètres spatio-temporels")
    if step_mean is not None:
        st.write(f"**Longueur de pas moyenne :** {step_mean:.1f} cm")
        st.write(f"**Variabilité (±1σ) :** {step_std:.1f} cm")
        if stepG_cm is not None and stepD_cm is not None:
            st.write(f"**Pas G :** {stepG_cm:.1f} cm — **Pas D :** {stepD_cm:.1f} cm")
        if step_asym is not None:
            st.write(f"**Asymétrie pas (G/D) :** {step_asym:.1f} %")
        st.caption("Estimation monocaméra 2D sans calibration métrique (échelle basée sur la taille).")
    else:
        st.warning("Longueur de pas non calculable.")

    st.subheader("⏱️ Temps du pas")
    col1, col2 = st.columns(2)

    with col1:
        if step_time_G_mean is not None:
            st.write(f"**Temps du pas Gauche :** {step_time_G_mean:.2f} s")
            st.write(f"**Variabilité Gauche :** ± {step_time_G_std:.2f} s")
            st.write(f"**Nombre de contacts Gauche :** {len(contactsG)}")
        else:
            st.write("**Temps du pas Gauche :** non calculable")

    with col2:
        if step_time_D_mean is not None:
            st.write(f"**Temps du pas Droit :** {step_time_D_mean:.2f} s")
            st.write(f"**Variabilité Droit :** ± {step_time_D_std:.2f} s")
            st.write(f"**Nombre de contacts Droit :** {len(contactsD)}")
        else:
            st.write("**Temps du pas Droit :** non calculable")

    st.caption("Les contacts au sol sont estimés à partir des minima verticaux des talons.")

    keyframe_path = os.path.join(tempfile.gettempdir(), "keyframe.png")
    cv2.imwrite(keyframe_path, frames[len(frames) // 2])

    figures = {}
    table_data = []
    asym_rows = []

    # ----- TRONC (une seule courbe) -----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [2, 1]})

    trunk_raw = np.array(data["Tronc"], dtype=float)
    trunk = smooth_clinical(trunk_raw, smooth_level=smooth)

    ax1.plot(trunk, label="Tronc", color="darkorange")
    for c0, c1, col in phases:
        ax1.axvspan(c0, c1, color=col, alpha=0.3)
    ax1.set_title("Tronc – Analyse")
    ax1.legend()

    if show_norm:
        norm = norm_curve("Tronc", len(trunk))
        norm = smooth_ma(norm, win=norm_smooth_win)
        ax2.plot(norm, color="green")
        ax2.set_title("Norme (lissée)" if norm_smooth_win and norm_smooth_win > 1 else "Norme")
    else:
        ax2.axis("off")

    st.pyplot(fig)

    fig_path = os.path.join(tempfile.gettempdir(), "Tronc_plot.png")
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    figures["Tronc"] = fig_path

    mask = ~np.isnan(trunk_raw)
    if mask.sum() == 0:
        tmin, tmean, tmax = np.nan, np.nan, np.nan
    else:
        vals = trunk[mask]
        tmin, tmean, tmax = float(np.min(vals)), float(np.mean(vals)), float(np.max(vals))

    table_data.append(["Tronc", f"{tmin:.1f}", f"{tmean:.1f}", f"{tmax:.1f}"])

    # ----- ARTICULATIONS droite/gauche -----
    for joint in ["Hanche", "Genou", "Cheville"]:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [2, 1]})

        g_raw = np.array(data[f"{joint} G"], dtype=float)
        d_raw = np.array(data[f"{joint} D"], dtype=float)

        g = smooth_clinical(g_raw, smooth_level=smooth)
        d = smooth_clinical(d_raw, smooth_level=smooth)

        ax1.plot(g, label="Gauche", color="red")
        ax1.plot(d, label="Droite", color="blue")
        for c0, c1, col in phases:
            ax1.axvspan(c0, c1, color=col, alpha=0.3)
        ax1.set_title(f"{joint} – Analyse")
        ax1.legend()

        if show_norm:
            norm = norm_curve(joint, len(g))
            norm = smooth_ma(norm, win=norm_smooth_win)
            ax2.plot(norm, color="green")
            ax2.set_title("Norme (lissée)" if norm_smooth_win and norm_smooth_win > 1 else "Norme")
        else:
            ax2.axis("off")

        st.pyplot(fig)

        fig_path = os.path.join(tempfile.gettempdir(), f"{joint}_plot.png")
        fig.savefig(fig_path, bbox_inches="tight")
        plt.close(fig)
        figures[joint] = fig_path

        def stats(arr_filtered, arr_raw):
            mask = ~np.isnan(arr_raw)
            if mask.sum() == 0:
                return np.nan, np.nan, np.nan, None
            vals = arr_filtered[mask]
            return float(np.min(vals)), float(np.mean(vals)), float(np.max(vals)), float(np.mean(vals))

        gmin, gmean, gmax, gmean_only = stats(g, g_raw)
        dmin, dmean, dmax, dmean_only = stats(d, d_raw)

        table_data.append([f"{joint} Gauche", f"{gmin:.1f}", f"{gmean:.1f}", f"{gmax:.1f}"])
        table_data.append([f"{joint} Droite", f"{dmin:.1f}", f"{dmean:.1f}", f"{dmax:.1f}"])

        a = asym_percent(gmean_only, dmean_only)
        if a is None:
            asym_rows.append([
                joint,
                f"{gmean_only:.1f}" if gmean_only is not None else "NA",
                f"{dmean_only:.1f}" if dmean_only is not None else "NA",
                "NA"
            ])
        else:
            asym_rows.append([joint, f"{gmean_only:.1f}", f"{dmean_only:.1f}", f"{a:.1f}"])

    st.subheader("↔️ Asymétries droite/gauche (angles)")
    for row in asym_rows:
        st.write(f"**{row[0]}** — Moy G: {row[1]}° | Moy D: {row[2]}° | Asym: {row[3]}%")

    st.subheader("🦶 Contacts au sol (talons)")
    fig_contact, ax = plt.subplots(figsize=(12, 4))

    x = np.arange(len(heelG_s)) / FPS
    ax.plot(x, heelG_s, label="Talon Gauche", color="red")
    ax.plot(x, heelD_s, label="Talon Droit", color="blue")

    if len(contactsG) > 0:
        ax.plot(contactsG / FPS, heelG_s[contactsG], "o", color="red")
    for c in contactsG:
        ax.axvline(c / FPS, color="red", alpha=0.15)

    if len(contactsD) > 0:
        ax.plot(contactsD / FPS, heelD_s[contactsD], "o", color="blue")
    for c in contactsD:
        ax.axvline(c / FPS, color="blue", alpha=0.15)

    ax.set_title("Détection des contacts au sol")
    ax.set_xlabel("Temps (s)")
    ax.legend()
    st.pyplot(fig_contact)

    contact_fig_path = os.path.join(tempfile.gettempdir(), "contacts_sol.png")
    fig_contact.savefig(contact_fig_path, bbox_inches="tight")
    plt.close(fig_contact)

    st.subheader("📸 Captures annotées (angles)")
    num_photos = st.slider("Nombre d'images extraites", 1, 10, 3)
    total_frames = len(frames)
    idxs = np.linspace(0, total_frames - 1, num_photos, dtype=int)

    annotated_images = []
    for i, idx in enumerate(idxs):
        frame = frames[idx]
        kp = detect_pose(frame)
        ann = annotate_frame(frame, kp, conf=conf)

        out_img = os.path.join(tempfile.gettempdir(), f"annotated_{i}.png")
        cv2.imwrite(out_img, ann)
        annotated_images.append(out_img)

        st.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB), caption=f"Image annotée {i+1} (frame {idx})")

    step_info = None
    if step_mean is not None:
        step_info = {"mean": step_mean, "std": step_std, "G": stepG_cm, "D": stepD_cm, "asym": step_asym}

    temporal_info = {
        "G_mean": step_time_G_mean,
        "G_std": step_time_G_std,
        "D_mean": step_time_D_mean,
        "D_std": step_time_D_std,
        "nG": len(contactsG),
        "nD": len(contactsD),
    }

    pdf_path = export_pdf(
        patient={
            "nom": nom,
            "prenom": prenom,
            "camera": camera_pos,
            "phase": phase_cote,
            "taille_cm": int(taille_cm),
            "show_norm": bool(show_norm)
        },
        keyframe_path=keyframe_path,
        figures=figures,
        table_data=table_data,
        annotated_images=annotated_images,
        step_info=step_info,
        asym_table=asym_rows,
        temporal_info=temporal_info,
        contact_fig_path=contact_fig_path
    )

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    st.success("✅ Rapport généré")
    st.download_button(
        "📄 Télécharger le rapport PDF",
        data=pdf_bytes,
        file_name=f"GaitScan_{nom}_{prenom}.pdf",
        mime="application/pdf"
    )
"""

POSTURE_FRONTALE_CODE = r"""
import streamlit as st
st.set_page_config(page_title="Analyseur Postural Pro (MediaPipe)", layout="wide")

import os
import tempfile
import numpy as np
import cv2
from PIL import Image
import math
from fpdf import FPDF
from datetime import datetime
import io

import mediapipe as mp
from streamlit_image_coordinates import streamlit_image_coordinates

st.title("🧍 Analyseur Postural Pro (MediaPipe)")
st.markdown("---")

# =========================
# 1) MEDIAPIPE
# =========================
mp_pose = mp.solutions.pose

@st.cache_resource
def load_pose():
    return mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

pose = load_pose()

# =========================
# 2) OUTILS
# =========================
def rotate_if_landscape(img_np_rgb: np.ndarray) -> np.ndarray:
    if img_np_rgb.shape[1] > img_np_rgb.shape[0]:
        img_np_rgb = cv2.rotate(img_np_rgb, cv2.ROTATE_90_CLOCKWISE)
    return img_np_rgb

def ensure_uint8_rgb(img: np.ndarray) -> np.ndarray:
    """Force image RGB uint8 contiguë."""
    if img is None:
        return None
    if img.dtype != np.uint8:
        img = img.astype(np.float32)
        if img.max() <= 1.5:
            img = img * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
    if not img.flags["C_CONTIGUOUS"]:
        img = np.ascontiguousarray(img)
    return img

def to_png_bytes(img_rgb_uint8: np.ndarray) -> bytes:
    """Encode en PNG bytes (ultra robuste)."""
    img_rgb_uint8 = ensure_uint8_rgb(img_rgb_uint8)
    pil = Image.fromarray(img_rgb_uint8, mode="RGB")
    bio = io.BytesIO()
    pil.save(bio, format="PNG")
    return bio.getvalue()

def calculate_angle(p1, p2, p3) -> float:
    v1 = np.array([p1[0]-p2[0], p1[1]-p2[1]], dtype=float)
    v2 = np.array([p3[0]-p2[0], p3[1]-p2[1]], dtype=float)
    dot = float(np.dot(v1, v2))
    mag = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if mag == 0:
        return 0.0
    return math.degrees(math.acos(np.clip(dot / mag, -1, 1)))

def femur_tibia_knee_angle(hip, knee, ankle) -> float:
    return calculate_angle(hip, knee, ankle)

def tibia_rearfoot_ankle_angle(knee, ankle, heel) -> float:
    return calculate_angle(knee, ankle, heel)

def pdf_safe(text) -> str:
    if text is None:
        return ""
    s = str(text)
    s = (s.replace("°", " deg")
           .replace("–", "-")
           .replace("—", "-")
           .replace("’", "'")
           .replace("“", '"')
           .replace("”", '"'))
    return s.encode("latin-1", errors="ignore").decode("latin-1")

def crop_to_landmarks(img_rgb_uint8: np.ndarray, res_pose, pad_ratio: float = 0.18) -> np.ndarray:
    """Cadrage auto autour du corps à partir des landmarks MediaPipe."""
    if res_pose is None or not res_pose.pose_landmarks:
        return img_rgb_uint8

    h, w = img_rgb_uint8.shape[:2]
    xs, ys = [], []
    for lm in res_pose.pose_landmarks.landmark:
        if lm.visibility < 0.2:
            continue
        xs.append(lm.x * w)
        ys.append(lm.y * h)

    if not xs or not ys:
        return img_rgb_uint8

    x1, x2 = max(0, int(min(xs))), min(w-1, int(max(xs)))
    y1, y2 = max(0, int(min(ys))), min(h-1, int(max(ys)))

    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    pad_x = int(bw * pad_ratio)
    pad_y = int(bh * pad_ratio)

    x1 = max(0, x1 - pad_x)
    x2 = min(w-1, x2 + pad_x)
    y1 = max(0, y1 - pad_y)
    y2 = min(h-1, y2 + pad_y)

    if x2 <= x1 or y2 <= y1:
        return img_rgb_uint8

    return img_rgb_uint8[y1:y2, x1:x2].copy()

def _to_float(val):
    if val is None:
        return None
    s = str(val).replace(",", ".")
    num = ""
    for ch in s:
        if ch.isdigit() or ch in ".-":
            num += ch
        elif num:
            break
    try:
        return float(num)
    except:
        return None

def _badge(status: str):
    if status == "OK":
        return "🟢 OK"
    if status == "SURV":
        return "🟠 À surveiller"
    return "🔴 À corriger"

def _status_from_mm(mm: float):
    if mm is None:
        return "SURV"
    if mm < 5:
        return "OK"
    if mm < 10:
        return "SURV"
    return "ALERTE"

def _status_from_deg(deg: float):
    if deg is None:
        return "SURV"
    if deg < 2:
        return "OK"
    if deg < 5:
        return "SURV"
    return "ALERTE"

# =========================
# PDF PRO (EN MÉMOIRE + COMPAT FPDF)
# =========================
def generate_pdf(data: dict, img_rgb_uint8: np.ndarray) -> bytes:
    from fpdf import FPDF
    import os
    import tempfile
    from PIL import Image
    from datetime import datetime

    def _pdf_safe(text):
        if text is None:
            return ""
        s = str(text)
        s = (s.replace("°", " deg")
               .replace("–", "-")
               .replace("—", "-")
               .replace("’", "'")
               .replace("“", '"')
               .replace("”", '"')
               .replace("\xa0", " "))
        return s.encode("latin-1", errors="ignore").decode("latin-1")

    def _to_float(val):
        try:
            s = str(val).replace(",", ".")
            num = ""
            for ch in s:
                if ch.isdigit() or ch in ".-":
                    num += ch
                elif num:
                    break
            return float(num)
        except:
            return None

    def _status_mm(v):
        if v is None:
            return "A SURV"
        if v < 5:
            return "OK"
        if v < 10:
            return "A SURV"
        return "ALERTE"

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Bandeau
    pdf.set_fill_color(31, 73, 125)
    pdf.rect(0, 0, 210, 35, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 20)
    pdf.set_y(10)
    pdf.cell(0, 10, "COMPTE-RENDU POSTURAL (IA)", align="C", ln=True)

    # Infos patient
    pdf.set_text_color(0, 0, 0)
    pdf.set_y(42)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(120, 8, _pdf_safe(f"Patient : {data.get('Nom','')}"), ln=0)
    pdf.set_font("Arial", '', 11)
    pdf.cell(70, 8, datetime.now().strftime("%d/%m/%Y %H:%M"), ln=1, align="R")
    pdf.ln(2)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    # ================= IMAGE (FIT DANS LA PAGE + PLUS PETITE + CENTRÉE) =================
    tmp_img = os.path.join(tempfile.gettempdir(), "posture_tmp.png")
    Image.fromarray(img_rgb_uint8).save(tmp_img)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 7, "Photographie annotée", ln=True)

    # Position actuelle et espace disponible
    y = pdf.get_y()
    page_w = pdf.w - 2 * pdf.l_margin
    avail_h = pdf.h - pdf.b_margin - y

    # Ratio image
    ih, iw = img_rgb_uint8.shape[:2]
    aspect = iw / ih  # largeur/hauteur

    # Largeur cible "plus petite"
    target_w = page_w * 0.62   # <-- diminue (0.5 / 0.45) si tu veux encore plus petit
    target_h = target_w / aspect

    # S'assurer que ça rentre verticalement (sinon FPDF pousse sur page suivante => page blanche)
    if target_h > avail_h:
        target_h = avail_h
        target_w = target_h * aspect

    x = (pdf.w - target_w) / 2
    pdf.image(tmp_img, x=x, y=y, w=target_w, h=target_h)
    pdf.set_y(y + target_h + 4)

    # Synthèse
    sh_mm = _to_float(data.get("Dénivelé Épaules (mm)"))
    hip_mm = _to_float(data.get("Dénivelé Bassin (mm)"))

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 7, "Synthèse", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.cell(0, 6, _pdf_safe(f"- Épaules : {data.get('Dénivelé Épaules (mm)','—')} [{_status_mm(sh_mm)}]"), ln=True)
    pdf.cell(0, 6, _pdf_safe(f"- Bassin  : {data.get('Dénivelé Bassin (mm)','—')} [{_status_mm(hip_mm)}]"), ln=True)
    pdf.ln(3)

    # Tableau
    pdf.set_font("Arial", 'B', 12)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(120, 9, "Indicateur", 1, 0, 'L', True)
    pdf.cell(70, 9, "Valeur", 1, 1, 'C', True)

    pdf.set_font("Arial", '', 11)
    for k, v in data.items():
        if k != "Nom":
            pdf.cell(120, 8, _pdf_safe(k), 1, 0, 'L')
            pdf.cell(70, 8, _pdf_safe(v), 1, 1, 'C')

    # Observations
    pdf.ln(4)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 7, "Observations automatiques", ln=True)
    pdf.set_font("Arial", '', 11)

    obs = []
    obs.append("Analyse générée automatiquement à partir d'une image 2D.")
    obs.append("Les mesures dépendent de la qualité de la prise de vue.")

    for o in obs:
        pdf.multi_cell(190, 6, _pdf_safe(f"- {o}"))

    # Footer
    pdf.set_y(-18)
    pdf.set_font("Arial", 'I', 8)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 10, "Document indicatif - Ne remplace pas un avis médical.", align="C")

    if os.path.exists(tmp_img):
        try:
            os.remove(tmp_img)
        except Exception:
            pass

    out = pdf.output(dest="S")
    if isinstance(out, (bytes, bytearray)):
        return bytes(out)
    return out.encode("latin-1")

# =========================
# POINTS ORIGINE + PREVIEW
# =========================
def extract_origin_points_from_mediapipe(img_rgb_uint8: np.ndarray):
    res = pose.process(img_rgb_uint8)
    if not res.pose_landmarks:
        return {}
    lm = res.pose_landmarks.landmark
    L = mp_pose.PoseLandmark
    h, w = img_rgb_uint8.shape[:2]

    def pt_px(enum_):
        p = lm[enum_.value]
        return (float(p.x * w), float(p.y * h))

    return {
        "Genou G": pt_px(L.LEFT_KNEE),
        "Genou D": pt_px(L.RIGHT_KNEE),
        "Cheville G": pt_px(L.LEFT_ANKLE),
        "Cheville D": pt_px(L.RIGHT_ANKLE),
        "Talon G": pt_px(L.LEFT_HEEL),
        "Talon D": pt_px(L.RIGHT_HEEL),

        "Hanche G": pt_px(L.LEFT_HIP),
        "Hanche D": pt_px(L.RIGHT_HIP),

        "_Epaule G": pt_px(L.LEFT_SHOULDER),
        "_Epaule D": pt_px(L.RIGHT_SHOULDER),
    }

def draw_preview(img_disp_rgb_uint8: np.ndarray, origin_points: dict, override_one: dict, scale: float) -> np.ndarray:
    out = img_disp_rgb_uint8.copy()
    out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    for name, p in origin_points.items():
        if name.startswith("_"):
            continue
        x = int(p[0] * scale)
        y = int(p[1] * scale)
        cv2.circle(out_bgr, (x, y), 6, (0, 255, 0), -1)

    for name, p in override_one.items():
        x = int(p[0] * scale)
        y = int(p[1] * scale)
        cv2.circle(out_bgr, (x, y), 10, (255, 0, 255), 3)
        cv2.line(out_bgr, (x - 12, y), (x + 12, y), (255, 0, 255), 2)
        cv2.line(out_bgr, (x, y - 12), (x, y + 12), (255, 0, 255), 2)
        cv2.putText(out_bgr, name, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)

    return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)

# =========================
# 3) SESSION STATE
# =========================
if "override_one" not in st.session_state:
    st.session_state["override_one"] = {}  # {"Cheville G": (x,y)}

# =========================
# 4) UI
# =========================
with st.sidebar:
    st.header("👤 Dossier Patient")
    nom = st.text_input("Nom complet", value="Anonyme")
    taille_cm = st.number_input("Taille (cm)", min_value=100, max_value=220, value=170)

    st.divider()
    source = st.radio("Source de l'image", ["📷 Caméra", "📁 Téléverser une photo"])

    st.divider()
    st.subheader("🖱️ Correction avant analyse")
    enable_click_edit = st.checkbox("Activer correction par clic", value=True)

    editable_points = ["Hanche G", "Hanche D", "Genou G", "Genou D", "Cheville G", "Cheville D", "Talon G", "Talon D"]
    point_to_edit = st.selectbox("Point à corriger", editable_points, disabled=not enable_click_edit)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("↩️ Reset point", disabled=not enable_click_edit):
            st.session_state["override_one"].pop(point_to_edit, None)
    with c2:
        if st.button("🧹 Reset tout", disabled=not enable_click_edit):
            st.session_state["override_one"] = {}

    st.divider()
    st.subheader("🖼️ Affichage")
    disp_w_user = st.slider("Largeur d'affichage (px)", min_value=320, max_value=900, value=520, step=10)
    auto_crop = st.checkbox("Cadrage automatique (autour du corps)", value=True)

col_input, col_result = st.columns([1, 1])

# =========================
# 5) INPUT IMAGE
# =========================
with col_input:
    if source == "📷 Caméra":
        image_data = st.camera_input("Capturez la posture")
    else:
        image_data = st.file_uploader("Format JPG/PNG", type=["jpg", "png", "jpeg"])

if not image_data:
    st.stop()

if isinstance(image_data, Image.Image):
    img_np = np.array(image_data.convert("RGB"))
else:
    img_np = np.array(Image.open(image_data).convert("RGB"))

img_np = rotate_if_landscape(img_np)
img_np = ensure_uint8_rgb(img_np)

# Cadrage auto (optionnel) sans casser le reste
res_for_crop = pose.process(img_np)
if auto_crop:
    img_np = crop_to_landmarks(img_np, res_for_crop, pad_ratio=0.18)
    img_np = ensure_uint8_rgb(img_np)

h, w = img_np.shape[:2]

# =========================
# 6) PREVIEW CLIQUABLE
# =========================
with col_input:
    st.subheader("📌 Cliquez pour placer le point sélectionné (avant analyse)")
    st.caption("Verts = points d'origine | Violet = point corrigé")

    disp_w = min(int(disp_w_user), w)
    scale = disp_w / w
    disp_h = int(h * scale)

    img_disp = cv2.resize(img_np, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
    img_disp = ensure_uint8_rgb(img_disp)

    origin_points = extract_origin_points_from_mediapipe(img_np)
    preview = draw_preview(img_disp, origin_points, st.session_state["override_one"], scale)

    coords = streamlit_image_coordinates(
        Image.open(io.BytesIO(to_png_bytes(preview))),
        key="img_click",
    )

    if enable_click_edit and coords is not None:
        cx = float(coords["x"])
        cy = float(coords["y"])
        x_orig = cx / scale
        y_orig = cy / scale
        st.session_state["override_one"][point_to_edit] = (x_orig, y_orig)
        st.success(f"✅ {point_to_edit} placé à ({x_orig:.0f}, {y_orig:.0f}) px")

    if st.session_state["override_one"]:
        st.write("**Point(s) corrigé(s) enregistré(s) :**")
        for k, (x, y) in st.session_state["override_one"].items():
            st.write(f"- {k} → ({x:.0f}, {y:.0f})")

# =========================
# 7) ANALYSE
# =========================
with col_result:
    st.subheader("⚙️ Analyse")
    run = st.button("▶ Lancer l'analyse")

if not run:
    st.stop()

with st.spinner("Détection (MediaPipe) + calculs..."):
    res = pose.process(img_np)
    if not res.pose_landmarks:
        st.error("Aucune pose détectée. Photo plus nette, en pied, bien centrée.")
        st.stop()

    lm = res.pose_landmarks.landmark
    L = mp_pose.PoseLandmark

    def pt(enum_):
        p = lm[enum_.value]
        return np.array([p.x * w, p.y * h], dtype=np.float32)

    LS = pt(L.LEFT_SHOULDER)
    RS = pt(L.RIGHT_SHOULDER)
    LH = pt(L.LEFT_HIP)
    RH = pt(L.RIGHT_HIP)
    LK = pt(L.LEFT_KNEE)
    RK = pt(L.RIGHT_KNEE)
    LA = pt(L.LEFT_ANKLE)
    RA = pt(L.RIGHT_ANKLE)
    LHE = pt(L.LEFT_HEEL)
    RHE = pt(L.RIGHT_HEEL)

    POINTS = {
        "Epaule G": LS, "Epaule D": RS,
        "Hanche G": LH, "Hanche D": RH,
        "Genou G": LK, "Genou D": RK,
        "Cheville G": LA, "Cheville D": RA,
        "Talon G": LHE, "Talon D": RHE,
    }

    for k, (x, y) in st.session_state["override_one"].items():
        if k in POINTS:
            POINTS[k] = np.array([x, y], dtype=np.float32)

    LS = POINTS["Epaule G"]; RS = POINTS["Epaule D"]
    LH = POINTS["Hanche G"]; RH = POINTS["Hanche D"]
    LK = POINTS["Genou G"];  RK = POINTS["Genou D"]
    LA = POINTS["Cheville G"]; RA = POINTS["Cheville D"]
    LHE = POINTS["Talon G"]; RHE = POINTS["Talon D"]

    raw_sh = math.degrees(math.atan2(LS[1]-RS[1], LS[0]-RS[0]))
    shoulder_angle = abs(raw_sh)
    if shoulder_angle > 90:
        shoulder_angle = abs(shoulder_angle - 180)

    raw_hip = math.degrees(math.atan2(LH[1]-RH[1], LH[0]-RH[0]))
    hip_angle = abs(raw_hip)
    if hip_angle > 90:
        hip_angle = abs(hip_angle - 180)

    knee_l = femur_tibia_knee_angle(LH, LK, LA)
    knee_r = femur_tibia_knee_angle(RH, RK, RA)
    ankle_l = tibia_rearfoot_ankle_angle(LK, LA, LHE)
    ankle_r = tibia_rearfoot_ankle_angle(RK, RA, RHE)

    px_height = max(LA[1], RA[1]) - min(LS[1], RS[1])
    mm_per_px = (float(taille_cm) * 10.0) / px_height if px_height > 0 else 0.0
    diff_shoulders_mm = abs(LS[1] - RS[1]) * mm_per_px
    diff_hips_mm = abs(LH[1] - RH[1]) * mm_per_px

    shoulder_lower = "Gauche" if LS[1] > RS[1] else "Droite"
    hip_lower = "Gauche" if LH[1] > RH[1] else "Droite"

    ann_bgr = cv2.cvtColor(img_np.copy(), cv2.COLOR_RGB2BGR)

    for _, p in POINTS.items():
        cv2.circle(ann_bgr, tuple(p.astype(int)), 7, (0, 255, 0), -1)

    for name in list(st.session_state["override_one"].keys()):
        if name in POINTS:
            p = POINTS[name]
            cv2.circle(ann_bgr, tuple(p.astype(int)), 14, (255, 0, 255), 3)
            cv2.putText(ann_bgr, name, (int(p[0]) + 10, int(p[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    cv2.line(ann_bgr, tuple(LS.astype(int)), tuple(RS.astype(int)), (255, 0, 0), 3)
    cv2.line(ann_bgr, tuple(LH.astype(int)), tuple(RH.astype(int)), (255, 0, 0), 3)

    annotated = cv2.cvtColor(ann_bgr, cv2.COLOR_BGR2RGB)
    annotated = ensure_uint8_rgb(annotated)

    results = {
        "Nom": nom,
        "Inclinaison Épaules (0=horizon)": f"{shoulder_angle:.1f}°",
        "Épaule la plus basse": shoulder_lower,
        "Dénivelé Épaules (mm)": f"{diff_shoulders_mm:.1f} mm",
        "Inclinaison Bassin (0=horizon)": f"{hip_angle:.1f}°",
        "Bassin le plus bas": hip_lower,
        "Dénivelé Bassin (mm)": f"{diff_hips_mm:.1f} mm",
        "Angle Genou Gauche (fémur-tibia)": f"{knee_l:.1f}°",
        "Angle Genou Droit (fémur-tibia)": f"{knee_r:.1f}°",
        "Angle Cheville G (tibia-arrière-pied)": f"{ankle_l:.1f}°",
        "Angle Cheville D (tibia-arrière-pied)": f"{ankle_r:.1f}°",
    }

# =========================
# 8) SORTIE (WEB + PDF)
# =========================
with col_result:
    st.subheader("🧾 Compte-rendu d'analyse posturale")

    sh_deg = _to_float(results.get("Inclinaison Épaules (0=horizon)"))
    hip_deg = _to_float(results.get("Inclinaison Bassin (0=horizon)"))
    sh_mm = _to_float(results.get("Dénivelé Épaules (mm)"))
    hip_mm = _to_float(results.get("Dénivelé Bassin (mm)"))

    st.markdown("### 🧑‍⚕️ Identité")
    st.write(f"**Patient :** {nom}")
    st.write(f"**Taille déclarée :** {taille_cm} cm")
    st.write(f"**Date/heure :** {datetime.now().strftime('%d/%m/%Y %H:%M')}")

    st.markdown("---")
    st.markdown("### 📌 Synthèse (mêmes données que le PDF)")

    sh_status = _status_from_mm(sh_mm)
    hip_status = _status_from_mm(hip_mm)
    sh_deg_status = _status_from_deg(sh_deg)
    hip_deg_status = _status_from_deg(hip_deg)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("**Épaules (mm)**")
        st.write(results.get("Dénivelé Épaules (mm)", "—"))
        st.write(_badge(sh_status))
    with c2:
        st.markdown("**Épaules (°)**")
        st.write(results.get("Inclinaison Épaules (0=horizon)", "—"))
        st.write(_badge(sh_deg_status))
    with c3:
        st.markdown("**Bassin (mm)**")
        st.write(results.get("Dénivelé Bassin (mm)", "—"))
        st.write(_badge(hip_status))
    with c4:
        st.markdown("**Bassin (°)**")
        st.write(results.get("Inclinaison Bassin (0=horizon)", "—"))
        st.write(_badge(hip_deg_status))

    st.markdown("### 🧩 Détails")
    left, right = st.columns(2)

    with left:
        st.markdown("**Alignement frontal**")
        st.write(f"- Inclinaison épaules : {results.get('Inclinaison Épaules (0=horizon)', '—')}")
        st.write(f"- Épaule la plus basse : {results.get('Épaule la plus basse', '—')}")
        st.write(f"- Dénivelé épaules : {results.get('Dénivelé Épaules (mm)', '—')}")
        st.write("")
        st.write(f"- Inclinaison bassin : {results.get('Inclinaison Bassin (0=horizon)', '—')}")
        st.write(f"- Bassin le plus bas : {results.get('Bassin le plus bas', '—')}")
        st.write(f"- Dénivelé bassin : {results.get('Dénivelé Bassin (mm)', '—')}")

    with right:
        st.markdown("**Membres inférieurs**")
        st.write(f"- Genou G (fémur-tibia) : {results.get('Angle Genou Gauche (fémur-tibia)', '—')}")
        st.write(f"- Genou D (fémur-tibia) : {results.get('Angle Genou Droit (fémur-tibia)', '—')}")
        st.write("")
        st.write(f"- Cheville G (tibia-arrière-pied) : {results.get('Angle Cheville G (tibia-arrière-pied)', '—')}")
        st.write(f"- Cheville D (tibia-arrière-pied) : {results.get('Angle Cheville D (tibia-arrière-pied)', '—')}")

    st.markdown("### ✅ Observations automatiques")
    obs = []
    if sh_status == "ALERTE" or sh_deg_status == "ALERTE":
        obs.append("Épaules : asymétrie marquée (contrôle clinique recommandé).")
    elif sh_status == "SURV" or sh_deg_status == "SURV":
        obs.append("Épaules : légère asymétrie (à surveiller).")
    else:
        obs.append("Épaules : alignement satisfaisant.")

    if hip_status == "ALERTE" or hip_deg_status == "ALERTE":
        obs.append("Bassin : bascule marquée (contrôle clinique recommandé).")
    elif hip_status == "SURV" or hip_deg_status == "SURV":
        obs.append("Bassin : légère bascule (à surveiller).")
    else:
        obs.append("Bassin : alignement satisfaisant.")

    for o in obs:
        st.write(f"- {o}")

    st.markdown("### 📝 Tableau des mesures (identique PDF)")
    st.table(results)

    st.markdown("### 🖼️ Image annotée")
    st.image(
        Image.fromarray(annotated, mode="RGB"),
        caption="Points verts = utilisés | Violet = corrigé",
        use_column_width=True
    )

    st.markdown("---")
    st.subheader("📄 PDF")
    pdf_bytes = generate_pdf(results, annotated)
    pdf_name = f"Bilan_{pdf_safe(results.get('Nom','Anonyme')).replace(' ', '_')}.pdf"
    st.download_button(
        label="📥 Télécharger le Bilan PDF",
        data=pdf_bytes,
        file_name=pdf_name,
        mime="application/pdf",
    )
"""

POSTURE_LATERALE_CODE = r"""
import streamlit as st
from fpdf import FPDF
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="Analyseur Postural Latéral", layout="wide")
st.title("🧍 Analyseur Postural Latéral (MediaPipe)")
st.markdown("Mesure sur **un seul côté visible** : jambe, cuisse, tronc et tête par rapport à la verticale.")
st.markdown("---")

mp_pose = mp.solutions.pose

@st.cache_resource
def load_pose():
    return mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

pose = load_pose()


def ensure_uint8_rgb(img: np.ndarray) -> np.ndarray:
    if img is None:
        return None
    if img.dtype != np.uint8:
        img = img.astype(np.float32)
        if img.max() <= 1.5:
            img = img * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
    if not img.flags["C_CONTIGUOUS"]:
        img = np.ascontiguousarray(img)
    return img


def rotate_if_landscape(img_np_rgb: np.ndarray) -> np.ndarray:
    if img_np_rgb.shape[1] > img_np_rgb.shape[0]:
        img_np_rgb = cv2.rotate(img_np_rgb, cv2.ROTATE_90_CLOCKWISE)
    return img_np_rgb


def to_png_bytes(img_rgb_uint8: np.ndarray) -> bytes:
    pil = Image.fromarray(ensure_uint8_rgb(img_rgb_uint8), mode="RGB")
    bio = io.BytesIO()
    pil.save(bio, format="PNG")
    return bio.getvalue()


def pdf_safe(text) -> str:
    if text is None:
        return ""
    s = str(text)
    s = (s.replace("°", " deg")
           .replace("–", "-")
           .replace("—", "-")
           .replace("’", "'")
           .replace("“", '"')
           .replace("”", '"'))
    return s.encode("latin-1", errors="ignore").decode("latin-1")


def _to_float(val):
    if val is None:
        return None
    s = str(val).replace(",", ".")
    num = ""
    for ch in s:
        if ch.isdigit() or ch in ".-":
            num += ch
        elif num:
            break
    try:
        return float(num)
    except Exception:
        return None


def _badge(status: str):
    if status == "OK":
        return "🟢 OK"
    if status == "SURV":
        return "🟠 À surveiller"
    return "🔴 À corriger"


def _status_from_deg(deg: float):
    if deg is None:
        return "SURV"
    if deg < 2:
        return "OK"
    if deg < 5:
        return "SURV"
    return "ALERTE"


def calculate_angle(p1, p2, p3) -> float:
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]], dtype=float)
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]], dtype=float)
    dot = float(np.dot(v1, v2))
    mag = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if mag == 0:
        return 0.0
    return math.degrees(math.acos(np.clip(dot / mag, -1, 1)))


def angle_segment_vs_vertical(p1, p2) -> float:
    dx = float(p2[0] - p1[0])
    dy = float(p2[1] - p1[1])
    return math.degrees(math.atan2(abs(dx), abs(dy) + 1e-9))


def signed_angle_segment_vs_vertical(p1, p2) -> float:
    dx = float(p2[0] - p1[0])
    dy = float(p2[1] - p1[1])
    return math.degrees(math.atan2(dx, abs(dy) + 1e-9))


def crop_to_landmarks(img_rgb_uint8: np.ndarray, res_pose, pad_ratio: float = 0.18) -> np.ndarray:
    if res_pose is None or not res_pose.pose_landmarks:
        return img_rgb_uint8
    h, w = img_rgb_uint8.shape[:2]
    xs, ys = [], []
    for lm in res_pose.pose_landmarks.landmark:
        if getattr(lm, "visibility", 1.0) < 0.2:
            continue
        xs.append(lm.x * w)
        ys.append(lm.y * h)
    if not xs or not ys:
        return img_rgb_uint8
    x1, x2 = max(0, int(min(xs))), min(w - 1, int(max(xs)))
    y1, y2 = max(0, int(min(ys))), min(h - 1, int(max(ys)))
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    pad_x = int(bw * pad_ratio)
    pad_y = int(bh * pad_ratio)
    x1 = max(0, x1 - pad_x)
    x2 = min(w - 1, x2 + pad_x)
    y1 = max(0, y1 - pad_y)
    y2 = min(h - 1, y2 + pad_y)
    if x2 <= x1 or y2 <= y1:
        return img_rgb_uint8
    return img_rgb_uint8[y1:y2, x1:x2].copy()


def choose_visible_side(landmarks):
    L = mp_pose.PoseLandmark
    left_ids = [L.LEFT_SHOULDER, L.LEFT_HIP, L.LEFT_KNEE, L.LEFT_ANKLE, L.LEFT_HEEL, L.LEFT_EAR]
    right_ids = [L.RIGHT_SHOULDER, L.RIGHT_HIP, L.RIGHT_KNEE, L.RIGHT_ANKLE, L.RIGHT_HEEL, L.RIGHT_EAR]

    left_score = sum(float(getattr(landmarks[i.value], "visibility", 0.0)) for i in left_ids)
    right_score = sum(float(getattr(landmarks[i.value], "visibility", 0.0)) for i in right_ids)
    return "left" if left_score >= right_score else "right"


def extract_points(img_rgb_uint8: np.ndarray):
    res = pose.process(img_rgb_uint8)
    if not res.pose_landmarks:
        return None, None
    lm = res.pose_landmarks.landmark
    side = choose_visible_side(lm)
    h, w = img_rgb_uint8.shape[:2]
    L = mp_pose.PoseLandmark

    def pt(enum_):
        p = lm[enum_.value]
        return np.array([p.x * w, p.y * h], dtype=np.float32)

    side_map = {
        "left": {
            "Epaule": pt(L.LEFT_SHOULDER),
            "Hanche": pt(L.LEFT_HIP),
            "Genou": pt(L.LEFT_KNEE),
            "Cheville": pt(L.LEFT_ANKLE),
            "Talon": pt(L.LEFT_HEEL),
            "Oreille": pt(L.LEFT_EAR),
        },
        "right": {
            "Epaule": pt(L.RIGHT_SHOULDER),
            "Hanche": pt(L.RIGHT_HIP),
            "Genou": pt(L.RIGHT_KNEE),
            "Cheville": pt(L.RIGHT_ANKLE),
            "Talon": pt(L.RIGHT_HEEL),
            "Oreille": pt(L.RIGHT_EAR),
        },
    }
    points = side_map[side]
    points["Nez"] = pt(L.NOSE)
    return side, points


def draw_preview(img_disp_rgb_uint8: np.ndarray, origin_points: dict, override_points: dict, scale: float) -> np.ndarray:
    out_bgr = cv2.cvtColor(img_disp_rgb_uint8.copy(), cv2.COLOR_RGB2BGR)
    for name, p in origin_points.items():
        x = int(p[0] * scale)
        y = int(p[1] * scale)
        cv2.circle(out_bgr, (x, y), 6, (0, 255, 0), -1)
        cv2.putText(out_bgr, name, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    for name, p in override_points.items():
        x = int(p[0] * scale)
        y = int(p[1] * scale)
        cv2.circle(out_bgr, (x, y), 10, (255, 0, 255), 3)
        cv2.line(out_bgr, (x - 12, y), (x + 12, y), (255, 0, 255), 2)
        cv2.line(out_bgr, (x, y - 12), (x, y + 12), (255, 0, 255), 2)
    return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)


def generate_pdf(data: dict, img_rgb_uint8: np.ndarray) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_fill_color(31, 73, 125)
    pdf.rect(0, 0, 210, 35, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 18)
    pdf.set_y(10)
    pdf.cell(0, 10, "COMPTE-RENDU POSTURAL LATERAL", align="C", ln=True)

    pdf.set_text_color(0, 0, 0)
    pdf.set_y(42)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(120, 8, pdf_safe(f"Patient : {data.get('Nom', '')}"), ln=0)
    pdf.set_font("Arial", '', 11)
    pdf.cell(70, 8, datetime.now().strftime("%d/%m/%Y %H:%M"), ln=1, align="R")
    pdf.ln(2)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    tmp_img = os.path.join(tempfile.gettempdir(), "posture_lateral_tmp.png")
    Image.fromarray(img_rgb_uint8).save(tmp_img)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 7, "Photographie annotée", ln=True)
    y = pdf.get_y()
    page_w = pdf.w - 2 * pdf.l_margin
    avail_h = pdf.h - pdf.b_margin - y
    ih, iw = img_rgb_uint8.shape[:2]
    aspect = iw / ih
    target_w = page_w * 0.62
    target_h = target_w / aspect
    if target_h > avail_h:
        target_h = avail_h
        target_w = target_h * aspect
    x = (pdf.w - target_w) / 2
    pdf.image(tmp_img, x=x, y=y, w=target_w, h=target_h)
    pdf.set_y(y + target_h + 4)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 7, "Synthèse", ln=True)
    pdf.set_font("Arial", '', 11)
    for key in [
        "Côté détecté",
        "Inclinaison Jambe / verticale",
        "Inclinaison Cuisse / verticale",
        "Inclinaison Tronc / verticale",
        "Inclinaison Tête-Cou / verticale",
    ]:
        if key in data:
            pdf.cell(0, 6, pdf_safe(f"- {key} : {data[key]}"), ln=True)

    pdf.ln(3)
    pdf.set_font("Arial", 'B', 12)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(120, 9, "Indicateur", 1, 0, 'L', True)
    pdf.cell(70, 9, "Valeur", 1, 1, 'C', True)
    pdf.set_font("Arial", '', 11)
    for k, v in data.items():
        if k != "Nom":
            pdf.cell(120, 8, pdf_safe(k), 1, 0, 'L')
            pdf.cell(70, 8, pdf_safe(v), 1, 1, 'C')

    pdf.set_y(-18)
    pdf.set_font("Arial", 'I', 8)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 10, "Document indicatif - Ne remplace pas un avis médical.", align="C")

    try:
        os.remove(tmp_img)
    except Exception:
        pass

    out = pdf.output(dest="S")
    if isinstance(out, (bytes, bytearray)):
        return bytes(out)
    return out.encode("latin-1")


if "override_points" not in st.session_state:
    st.session_state["override_points"] = {}

with st.sidebar:
    st.header("👤 Dossier patient")
    nom = st.text_input("Nom complet", value="Anonyme")
    taille_cm = st.number_input("Taille (cm)", min_value=100, max_value=230, value=170)
    source = st.radio("Source de l'image", ["📷 Caméra", "📁 Téléverser une photo"])
    st.divider()
    st.subheader("🖱️ Correction avant analyse")
    enable_click_edit = st.checkbox("Activer correction par clic", value=True)
    point_to_edit = st.selectbox(
        "Point à corriger",
        ["Epaule", "Hanche", "Genou", "Cheville", "Talon", "Oreille", "Nez"],
        disabled=not enable_click_edit,
    )
    c1, c2 = st.columns(2)
    with c1:
        if st.button("↩️ Reset point", disabled=not enable_click_edit):
            st.session_state["override_points"].pop(point_to_edit, None)
    with c2:
        if st.button("🧹 Reset tout", disabled=not enable_click_edit):
            st.session_state["override_points"] = {}
    st.divider()
    disp_w_user = st.slider("Largeur d'affichage (px)", 320, 900, 520, 10)
    auto_crop = st.checkbox("Cadrage automatique", value=True)

col_input, col_result = st.columns([1, 1])

with col_input:
    if source == "📷 Caméra":
        image_data = st.camera_input("Capturez la posture latérale")
    else:
        image_data = st.file_uploader("Format JPG/PNG", type=["jpg", "jpeg", "png"])

if not image_data:
    st.stop()

if isinstance(image_data, Image.Image):
    img_np = np.array(image_data.convert("RGB"))
else:
    img_np = np.array(Image.open(image_data).convert("RGB"))

img_np = ensure_uint8_rgb(rotate_if_landscape(img_np))
res_for_crop = pose.process(img_np)
if auto_crop:
    img_np = ensure_uint8_rgb(crop_to_landmarks(img_np, res_for_crop, pad_ratio=0.18))

h, w = img_np.shape[:2]
side_detected, origin_points = extract_points(img_np)
if origin_points is None:
    st.error("Aucune pose détectée. Utilisez une photo nette, de profil, en pied.")
    st.stop()

with col_input:
    st.subheader("📌 Cliquez pour corriger le point sélectionné")
    st.caption(f"Côté détecté automatiquement : **{'Gauche' if side_detected == 'left' else 'Droite'}**")
    disp_w = min(int(disp_w_user), w)
    scale = disp_w / w
    disp_h = int(h * scale)
    img_disp = cv2.resize(img_np, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
    preview = draw_preview(img_disp, origin_points, st.session_state["override_points"], scale)
    coords = streamlit_image_coordinates(Image.open(io.BytesIO(to_png_bytes(preview))), key="img_click")
    if enable_click_edit and coords is not None:
        x_orig = float(coords["x"]) / scale
        y_orig = float(coords["y"]) / scale
        st.session_state["override_points"][point_to_edit] = (x_orig, y_orig)
        st.success(f"✅ {point_to_edit} placé à ({x_orig:.0f}, {y_orig:.0f}) px")
    if st.session_state["override_points"]:
        st.write("**Point(s) corrigé(s) :**")
        for k, (x, y) in st.session_state["override_points"].items():
            st.write(f"- {k} → ({x:.0f}, {y:.0f})")

with col_result:
    st.subheader("⚙️ Analyse")
    run = st.button("▶ Lancer l'analyse")

if not run:
    st.stop()

points = {k: v.copy() for k, v in origin_points.items()}
for k, v in st.session_state["override_points"].items():
    if k in points:
        points[k] = np.array([v[0], v[1]], dtype=np.float32)

Epaule = points["Epaule"]
Hanche = points["Hanche"]
Genou = points["Genou"]
Cheville = points["Cheville"]
Talon = points["Talon"]
Oreille = points["Oreille"]
Nez = points["Nez"]

incl_jambe = angle_segment_vs_vertical(Genou, Cheville)
incl_cuisse = angle_segment_vs_vertical(Hanche, Genou)
incl_tronc = angle_segment_vs_vertical(Hanche, Epaule)
incl_tete_cou = angle_segment_vs_vertical(Epaule, Oreille)
incl_tete_nez = angle_segment_vs_vertical(Oreille, Nez)

signed_tronc = signed_angle_segment_vs_vertical(Hanche, Epaule)
signed_tete = signed_angle_segment_vs_vertical(Epaule, Oreille)
angle_genou = calculate_angle(Hanche, Genou, Cheville)
angle_cheville = calculate_angle(Genou, Cheville, Talon)

sens_tronc = "Vers l'avant" if signed_tronc > 0 else "Vers l'arrière"
sens_tete = "Vers l'avant" if signed_tete > 0 else "Vers l'arrière"

results = {
    "Nom": nom,
    "Plan": "Latéral",
    "Côté détecté": "Gauche" if side_detected == "left" else "Droite",
    "Inclinaison Jambe / verticale": f"{incl_jambe:.1f}°",
    "Inclinaison Cuisse / verticale": f"{incl_cuisse:.1f}°",
    "Inclinaison Tronc / verticale": f"{incl_tronc:.1f}°",
    "Sens inclinaison tronc": sens_tronc,
    "Inclinaison Tête-Cou / verticale": f"{incl_tete_cou:.1f}°",
    "Sens inclinaison tête": sens_tete,
    "Inclinaison Tête (oreille-nez)": f"{incl_tete_nez:.1f}°",
    "Angle Genou": f"{angle_genou:.1f}°",
    "Angle Cheville": f"{angle_cheville:.1f}°",
}

ann_bgr = cv2.cvtColor(img_np.copy(), cv2.COLOR_RGB2BGR)
for _, p in points.items():
    cv2.circle(ann_bgr, tuple(np.round(p).astype(int)), 7, (0, 255, 0), -1)
for name, p in st.session_state["override_points"].items():
    arr = np.array(p)
    cv2.circle(ann_bgr, tuple(np.round(arr).astype(int)), 14, (255, 0, 255), 3)
    cv2.putText(ann_bgr, name, (int(arr[0]) + 8, int(arr[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
cv2.line(ann_bgr, tuple(np.round(Hanche).astype(int)), tuple(np.round(Epaule).astype(int)), (255, 0, 0), 3)
cv2.line(ann_bgr, tuple(np.round(Hanche).astype(int)), tuple(np.round(Genou).astype(int)), (0, 200, 0), 3)
cv2.line(ann_bgr, tuple(np.round(Genou).astype(int)), tuple(np.round(Cheville).astype(int)), (0, 255, 255), 3)
cv2.line(ann_bgr, tuple(np.round(Epaule).astype(int)), tuple(np.round(Oreille).astype(int)), (255, 0, 255), 3)
cv2.line(ann_bgr, tuple(np.round(Oreille).astype(int)), tuple(np.round(Nez).astype(int)), (100, 255, 100), 2)
annotated = cv2.cvtColor(ann_bgr, cv2.COLOR_BGR2RGB)
annotated = ensure_uint8_rgb(annotated)

with col_result:
    st.subheader("🧾 Compte-rendu d'analyse posturale")
    st.markdown("### 🧑‍⚕️ Identité")
    st.write(f"**Patient :** {nom}")
    st.write(f"**Taille déclarée :** {taille_cm} cm")
    st.write(f"**Date/heure :** {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    st.write(f"**Côté détecté :** {results['Côté détecté']}")
    st.markdown("---")

    leg_deg = _to_float(results.get("Inclinaison Jambe / verticale"))
    thigh_deg = _to_float(results.get("Inclinaison Cuisse / verticale"))
    trunk_deg = _to_float(results.get("Inclinaison Tronc / verticale"))
    head_deg = _to_float(results.get("Inclinaison Tête-Cou / verticale"))

    st.markdown("### 📌 Synthèse latérale")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("**Jambe / verticale**")
        st.write(results["Inclinaison Jambe / verticale"])
        st.write(_badge(_status_from_deg(leg_deg)))
    with c2:
        st.markdown("**Cuisse / verticale**")
        st.write(results["Inclinaison Cuisse / verticale"])
        st.write(_badge(_status_from_deg(thigh_deg)))
    with c3:
        st.markdown("**Tronc / verticale**")
        st.write(results["Inclinaison Tronc / verticale"])
        st.write(_badge(_status_from_deg(trunk_deg)))
    with c4:
        st.markdown("**Tête / verticale**")
        st.write(results["Inclinaison Tête-Cou / verticale"])
        st.write(_badge(_status_from_deg(head_deg)))

    st.markdown("### 🧩 Détails")
    for key in [
        "Inclinaison Jambe / verticale",
        "Inclinaison Cuisse / verticale",
        "Inclinaison Tronc / verticale",
        "Sens inclinaison tronc",
        "Inclinaison Tête-Cou / verticale",
        "Sens inclinaison tête",
        "Inclinaison Tête (oreille-nez)",
        "Angle Genou",
        "Angle Cheville",
    ]:
        st.write(f"- {key} : {results[key]}")

    st.markdown("### ✅ Observations automatiques")
    obs = []
    if trunk_deg is not None:
        obs.append("Tronc : alignement satisfaisant." if trunk_deg < 2 else "Tronc : légère inclinaison sagittale." if trunk_deg < 5 else "Tronc : inclinaison marquée.")
    if head_deg is not None:
        obs.append("Tête/cou : alignement satisfaisant." if head_deg < 2 else "Tête/cou : légère projection ou inclinaison." if head_deg < 5 else "Tête/cou : désalignement marqué.")
    if leg_deg is not None:
        obs.append("Jambe : orientation proche de la verticale." if leg_deg < 2 else "Jambe : légère inclinaison sagittale." if leg_deg < 5 else "Jambe : inclinaison marquée.")
    for o in obs:
        st.write(f"- {o}")

    st.markdown("### 📝 Tableau des mesures")
    st.table(results)

    st.markdown("### 🖼️ Image annotée")
    st.image(annotated, caption="Points verts = utilisés | Violet = corrigé", use_column_width=True)

    st.markdown("---")
    st.subheader("📄 PDF")
    pdf_bytes = generate_pdf(results, annotated)
    pdf_name = f"Bilan_Lateral_{pdf_safe(results.get('Nom', 'Anonyme')).replace(' ', '_')}.pdf"
    st.download_button(
        label="📥 Télécharger le Bilan PDF",
        data=pdf_bytes,
        file_name=pdf_name,
        mime="application/pdf",
    )
"""


# ============================================================
# OUTILS COMMUNS
# ============================================================
class _StopLegacy(Exception):
    pass


def uploaded_to_memory(uploaded_file):
    if uploaded_file is None:
        return None
    data = uploaded_file.getvalue()
    bio = io.BytesIO(data)
    bio.name = getattr(uploaded_file, "name", "input.bin")
    bio.type = getattr(uploaded_file, "type", None)
    bio.size = len(data)
    return bio


def pdf_safe(text: Any) -> str:
    if text is None:
        return ""
    s = str(text)
    replacements = {
        "€": "EUR", "°": " deg", "–": "-", "—": "-", "’": "'", "‘": "'",
        "“": '"', "”": '"', "…": "...", "é": "e", "è": "e", "ê": "e",
        "ë": "e", "à": "a", "â": "a", "ä": "a", "î": "i", "ï": "i",
        "ô": "o", "ö": "o", "ù": "u", "û": "u", "ü": "u", "ç": "c",
        "\xa0": " ",
    }
    for a, b in replacements.items():
        s = s.replace(a, b)
    s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def pdf_write(pdf: FPDF, text: Any) -> None:
    t = pdf_safe(text)
    if not t:
        return
    chunks = [t[i:i+140] for i in range(0, len(t), 140)] or [""]
    for chunk in chunks:
        try:
            pdf.multi_cell(0, 7, chunk)
        except Exception:
            safe = chunk.encode("latin-1", "ignore").decode("latin-1")
            if not safe.strip():
                safe = "[texte non exportable]"
            pdf.multi_cell(0, 7, safe)


def build_global_pdf(patient: Dict[str, Any], module_runs: List[Dict[str, Any]]) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Arial", "B", 15)
    pdf.cell(0, 10, "Compte-rendu global", ln=True)

    pdf.set_font("Arial", "", 11)
    pdf_write(pdf, f"Patient : {patient.get('nom', '')} {patient.get('prenom', '')}")
    pdf_write(pdf, f"Date : {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    pdf_write(pdf, f"Taille : {patient.get('taille_cm', '')} cm")
    pdf_write(pdf, f"Seuil confiance : {patient.get('conf', '')}")
    pdf_write(pdf, f"Lissage : {patient.get('smooth', '')}")
    pdf_write(pdf, f"Afficher la norme : {'Oui' if patient.get('show_norm') else 'Non'}")
    pdf_write(pdf, f"Angle de film : {patient.get('camera_pos', '')}")
    pdf_write(pdf, f"Phases : {patient.get('phase_cote', '')}")
    pdf.ln(3)

    for run in module_runs:
        pdf.set_font("Arial", "B", 12)
        pdf_write(pdf, run.get("title", "Module"))
        pdf.set_font("Arial", "", 11)
        pdf_write(pdf, f"Statut : {run.get('status', '')}")
        if run.get("input_name"):
            pdf_write(pdf, f"Fichier : {run['input_name']}")
        if run.get("downloads_count", 0):
            pdf_write(pdf, f"Fichiers produits : {run['downloads_count']}")
        if run.get("error"):
            pdf_write(pdf, "Erreur :")
            lines = str(run["error"]).splitlines()
            if lines:
                pdf_write(pdf, lines[-1])
        pdf.ln(2)

    return bytes(pdf.output(dest="S"))


# ============================================================
# PROXY STREAMLIT
# ============================================================
class LegacySidebarProxy:
    def __init__(self, owner, real_sidebar):
        self._owner = owner
        self._real = real_sidebar

    def __getattr__(self, name):
        return getattr(self._owner, name)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class LegacyStreamlit(types.ModuleType):
    def __init__(self, real_st, shared, module_title, uploaded_file, launch_analysis):
        super().__init__("streamlit")
        self._real = real_st
        self._shared = shared
        self._module_title = module_title
        self._uploaded_file = uploaded_file
        self._launch_analysis = launch_analysis
        self._downloads = []
        self._logs = []

        self.sidebar = LegacySidebarProxy(self, real_st.sidebar)
        self.session_state = real_st.session_state
        self.components = types.SimpleNamespace(v1=components)
        self.cache_data = getattr(real_st, "cache_data", lambda *a, **k: (lambda f: f))
        self.cache_resource = getattr(real_st, "cache_resource", lambda *a, **k: (lambda f: f))
        self.fragment = getattr(real_st, "fragment", lambda f=None, *a, **k: f if f else (lambda x: x))
        self.query_params = getattr(real_st, "query_params", None)

    def __getattr__(self, name):
        return getattr(self._real, name)

    def set_page_config(self, *args, **kwargs):
        return None

    def stop(self):
        raise _StopLegacy()

    def rerun(self):
        raise _StopLegacy()

    def _label(self, label) -> str:
        return str(label).strip().lower()

    def _log(self, prefix, value):
        txt = pdf_safe(value)
        if txt:
            self._logs.append(f"{prefix} {txt}")

    def write(self, *args, **kwargs):
        if args:
            self._log("WRITE:", " ".join(str(a) for a in args))
        return self._real.write(*args, **kwargs)

    def markdown(self, body, *args, **kwargs):
        self._log("MARKDOWN:", body)
        return self._real.markdown(body, *args, **kwargs)

    def success(self, body, *args, **kwargs):
        self._log("SUCCESS:", body)
        return self._real.success(body, *args, **kwargs)

    def info(self, body, *args, **kwargs):
        self._log("INFO:", body)
        return self._real.info(body, *args, **kwargs)

    def warning(self, body, *args, **kwargs):
        self._log("WARNING:", body)
        return self._real.warning(body, *args, **kwargs)

    def error(self, body, *args, **kwargs):
        self._log("ERROR:", body)
        return self._real.error(body, *args, **kwargs)

    def title(self, body, *args, **kwargs):
        self._log("TITLE:", body)
        return self._real.title(body, *args, **kwargs)

    def header(self, body, *args, **kwargs):
        self._log("HEADER:", body)
        return self._real.header(body, *args, **kwargs)

    def subheader(self, body, *args, **kwargs):
        self._log("SUBHEADER:", body)
        return self._real.subheader(body, *args, **kwargs)

    def caption(self, body, *args, **kwargs):
        self._log("CAPTION:", body)
        return self._real.caption(body, *args, **kwargs)

    def text_input(self, label, value="", *args, **kwargs):
        lk = self._label(label)
        if "nom complet" in lk:
            return self._shared.get("nom_complet", value)
        if lk == "nom":
            return self._shared.get("nom", value)
        if "prénom" in lk or "prenom" in lk:
            return self._shared.get("prenom", value)
        return self._real.text_input(label, value=value, *args, **kwargs)

    def number_input(self, label, *args, **kwargs):
        lk = self._label(label)
        if "taille" in lk:
            return self._shared.get("taille_cm", kwargs.get("value", 170))
        return self._real.number_input(label, *args, **kwargs)

    def slider(self, label, *args, **kwargs):
        lk = self._label(label)
        if "seuil confiance" in lk or lk == "confiance":
            return self._shared.get("conf", kwargs.get("value", 0.3))
        if "lissage norme" in lk:
            return self._shared.get("norm_smooth_win", kwargs.get("value", 7))
        if "lissage" in lk:
            return self._shared.get("smooth", kwargs.get("value", 3))
        if "nombre d'images" in lk:
            return self._shared.get("num_photos", kwargs.get("value", 3))
        return self._real.slider(label, *args, **kwargs)

    def checkbox(self, label, value=False, *args, **kwargs):
        lk = self._label(label)
        if "afficher la norme" in lk:
            return self._shared.get("show_norm", value)
        return self._real.checkbox(label, value=value, *args, **kwargs)

    def selectbox(self, label, options, *args, **kwargs):
        lk = self._label(label)
        if "angle de film" in lk:
            target = self._shared.get("camera_pos")
            if target in options:
                return target
        if "phases" in lk:
            target = self._shared.get("phase_cote")
            if target in options:
                return target
        return self._real.selectbox(label, options, *args, **kwargs)

    def button(self, label, *args, **kwargs):
        lk = self._label(label)
        if "lancer l'analyse" in lk or "analyser" in lk or "lancer analyse" in lk:
            return bool(self._launch_analysis)
        return self._real.button(label, *args, **kwargs)

    def file_uploader(self, label, *args, **kwargs):
        return uploaded_to_memory(self._uploaded_file)

    def camera_input(self, label, *args, **kwargs):
        return None

    def download_button(self, label, data=None, file_name=None, mime=None, *args, **kwargs):
        self._downloads.append({
            "label": label,
            "data": data,
            "file_name": file_name,
            "mime": mime,
            "module_title": self._module_title,
        })
        return self._real.download_button(label, data=data, file_name=file_name, mime=mime, *args, **kwargs)


@contextmanager
def patched_streamlit(fake_streamlit):
    old_streamlit = sys.modules.get("streamlit")
    old_components = sys.modules.get("streamlit.components")
    old_components_v1 = sys.modules.get("streamlit.components.v1")

    fake_components = types.ModuleType("streamlit.components")
    fake_components.v1 = components

    sys.modules["streamlit"] = fake_streamlit
    sys.modules["streamlit.components"] = fake_components
    sys.modules["streamlit.components.v1"] = components
    try:
        yield
    finally:
        if old_streamlit is not None:
            sys.modules["streamlit"] = old_streamlit
        else:
            sys.modules.pop("streamlit", None)
        if old_components is not None:
            sys.modules["streamlit.components"] = old_components
        else:
            sys.modules.pop("streamlit.components", None)
        if old_components_v1 is not None:
            sys.modules["streamlit.components.v1"] = old_components_v1
        else:
            sys.modules.pop("streamlit.components.v1", None)


def run_legacy_module(code_text: str, shared_values: Dict[str, Any], module_title: str, uploaded_file):
    fake_st = LegacyStreamlit(
        real_st=st,
        shared=shared_values,
        module_title=module_title,
        uploaded_file=uploaded_file,
        launch_analysis=True,
    )

    namespace = {
        "__name__": f"legacy_{re.sub(r'[^a-zA-Z0-9_]+', '_', module_title)}",
        "__file__": f"legacy_{re.sub(r'[^a-zA-Z0-9_]+', '_', module_title)}.py",
    }

    try:
        with patched_streamlit(fake_st):
            exec(code_text, namespace, namespace)
        return {"ok": True, "error": None, "downloads": fake_st._downloads, "logs": fake_st._logs}
    except _StopLegacy:
        return {"ok": True, "error": None, "downloads": fake_st._downloads, "logs": fake_st._logs}
    except Exception:
        return {"ok": False, "error": traceback.format_exc(), "downloads": fake_st._downloads, "logs": fake_st._logs}


# ============================================================
# APP COMMUNE
# ============================================================
st.set_page_config(page_title="Analyse biomécanique unifiée", layout="wide")
st.title("Analyse biomécanique unifiée")
st.caption("Une seule saisie des paramètres communs, avec les 4 codes d'origine conservés explicitement.")

with st.sidebar:
    st.header("Paramètres communs")
    nom = st.text_input("Nom", "DURAND")
    prenom = st.text_input("Prénom", "Jean")
    nom_complet = st.text_input("Nom complet", f"{nom} {prenom}")
    taille_cm = st.number_input("Taille du patient (cm)", min_value=80, max_value=230, value=170, step=1)
    conf = st.slider("Seuil confiance", 0.1, 0.9, 0.3, 0.05)
    smooth = st.slider("Lissage", 0, 10, 3)
    show_norm = st.checkbox("Afficher la norme", value=True)
    camera_pos = st.selectbox("Angle de film", ["Devant", "Derrière", "Profil droit", "Profil gauche"])
    phase_cote = st.selectbox("Phases", ["Aucune", "Droite", "Gauche", "Les deux"])
    norm_smooth_win = st.slider("Lissage norme (simple)", 1, 21, 7, 2)
    num_photos = st.slider("Nombre d'images extraites", 1, 10, 3)

shared_values = {
    "nom": nom,
    "prenom": prenom,
    "nom_complet": nom_complet,
    "taille_cm": int(taille_cm),
    "conf": conf,
    "smooth": smooth,
    "show_norm": bool(show_norm),
    "camera_pos": camera_pos,
    "phase_cote": phase_cote,
    "norm_smooth_win": norm_smooth_win,
    "num_photos": num_photos,
}

st.subheader("Fichiers d'analyse")
col1, col2 = st.columns(2)
with col1:
    file_front = st.file_uploader("Vidéo - analyse frontale / vue arrière", type=["mp4", "avi", "mov"], key="front")
    file_cine = st.file_uploader("Vidéo - analyse cinématique", type=["mp4", "avi", "mov"], key="cine")
with col2:
    file_post_front = st.file_uploader("Image - analyse posturale frontale", type=["png", "jpg", "jpeg"], key="post_front")
    file_post_lat = st.file_uploader("Image - analyse posturale latérale", type=["png", "jpg", "jpeg"], key="post_lat")

plan = [
    ("Analyse frontale / vue arrière", FRONTALE_CODE, file_front),
    ("Analyse cinématique", CINEMATIQUE_CODE, file_cine),
    ("Analyse posturale frontale", POSTURE_FRONTALE_CODE, file_post_front),
    ("Analyse posturale latérale", POSTURE_LATERALE_CODE, file_post_lat),
]

if st.button("▶ Lancer les analyses", type="primary"):
    module_runs = []

    for module_title, module_code, uploaded in plan:
        st.markdown("---")
        st.subheader(module_title)

        if uploaded is None:
            st.info("Non analysé")
            module_runs.append({
                "title": module_title,
                "status": "Non analysé",
                "input_name": None,
                "downloads_count": 0,
                "error": None,
            })
            continue

        result = run_legacy_module(
            code_text=module_code,
            shared_values=shared_values,
            module_title=module_title,
            uploaded_file=uploaded,
        )

        status = "Analysé" if result["ok"] else "Erreur"

        if result["ok"]:
            st.success("Analyse terminée")
        else:
            st.error("Erreur pendant l'exécution du module")
            st.code(result["error"])

        if result["downloads"]:
            st.caption("Fichiers produits par le module :")
            for i, dl in enumerate(result["downloads"], start=1):
                fname = dl.get("file_name") or f"sortie_{i}.bin"
                mime = dl.get("mime") or "application/octet-stream"
                st.download_button(
                    label=f"Télécharger - {module_title} - {fname}",
                    data=dl.get("data"),
                    file_name=fname,
                    mime=mime,
                    key=f"download_{module_title}_{i}",
                )

        module_runs.append({
            "title": module_title,
            "status": status,
            "input_name": getattr(uploaded, "name", None),
            "downloads_count": len(result["downloads"]),
            "error": result["error"],
        })

    st.markdown("---")
    st.subheader("Synthèse globale")
    for run in module_runs:
        st.write(f"- **{run['title']}** : {run['status']}")

    pdf_bytes = build_global_pdf(shared_values, module_runs)
    st.download_button(
        "📄 Télécharger le compte-rendu global PDF",
        data=pdf_bytes,
        file_name=f"Compte_rendu_global_{nom}_{prenom}.pdf",
        mime="application/pdf",
    )
else:
    st.info("Renseignez les paramètres communs, chargez les fichiers, puis lancez les analyses.")
