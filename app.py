"""Streamlit showcase interface for the Audio-Visual Road Accident Detection project.

Run from the project root:
    python -m streamlit run app.py
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import torch

from create_dataset import predict_video
from inspect_video import Config, probe

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "datasets" / "processed" / "models" / "model.pt"
UPLOAD_DIR = PROJECT_DIR / "datasets" / "processed" / "uploads"
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm", "m4v"}

st.set_page_config(
    page_title="Road Accident Detection | AIDL",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.8rem; padding-bottom: 2.5rem; max-width: 1200px;}
      .hero {padding: 1.35rem 1.5rem; border: 1px solid rgba(128,128,128,.25);
             border-radius: 16px; background: linear-gradient(120deg, rgba(30,90,130,.16), rgba(35,145,115,.10));}
      .muted {opacity: .72; font-size: .9rem;}
      .result {padding: 1rem 1.2rem; border-radius: 14px; border: 1px solid rgba(128,128,128,.3);}
    </style>
    """,
    unsafe_allow_html=True,
)


def _safe_filename(name: str) -> str:
    suffix = Path(name).suffix.lower().lstrip(".")
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported video extension: .{suffix or '(none)'}")
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).stem).strip("._") or "uploaded_video"
    return f"{stem[:70]}.{suffix}"


def _persist_upload(uploaded_file: Any) -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = _safe_filename(uploaded_file.name)
    payload = uploaded_file.getvalue()
    if not payload:
        raise ValueError("The uploaded file is empty.")
    digest = hashlib.sha256(payload).hexdigest()[:12]
    destination = UPLOAD_DIR / f"{digest}_{safe_name}"
    if not destination.exists():
        destination.write_bytes(payload)
    return destination


def _csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    if not rows:
        return b""
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


def _mode_description(result: dict[str, Any]) -> tuple[str, str]:
    mode = str(result.get("mode", "visual_only"))
    alpha = float(result.get("visual_weight", 1.0))
    if mode == "audio_visual":
        return "Audio + visual decision", f"Late fusion is active (visual weight {alpha:.2f}; audio weight {1-alpha:.2f})."
    if mode == "audio_only":
        return "Audio-only decision", "The validation-selected fusion weight assigns the decision to the audio branch."
    if result.get("audio_score") is not None and alpha >= 0.999:
        return "Visual-only decision", "The validation-selected visual weight is 1.00. The audio score is shown for diagnostics but does not affect the final label."
    return "Visual-only decision", "Audio was unavailable, unusable, or explicitly disabled for this prediction."


st.markdown(
    """
    <div class="hero">
      <h1 style="margin:0 0 .35rem 0">Audio-Visual Road Accident Detection</h1>
      <p style="margin:0">Deep-learning analysis of road videos using visual features and audio signals.</p>
      <p class="muted" style="margin:.6rem 0 0 0">Academic prototype · Upload a supported video to analyse it with the trained model.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.warning(
    "Research demonstration only. This prototype is not a certified safety system. "
    "Model scores are uncalibrated and can be wrong on unfamiliar footage."
)

with st.sidebar:
    st.header("Analysis settings")
    device_choice = st.selectbox("Processing device", ["auto", "cpu", "cuda"], index=0)
    visual_only = st.checkbox("Ignore audio (visual-only prediction)", value=False)
    st.caption("Use this for videos with music overlays, replaced audio, or unusable sound.")
    st.divider()
    st.subheader("Input limits")
    st.write("Formats: MP4, AVI, MOV, MKV, WEBM, M4V")
    st.write("Maximum duration: 120 seconds")
    st.write("The model analyses overlapping 4-second windows.")

if not MODEL_PATH.is_file():
    st.error("Trained model checkpoint not found.")
    st.code("python create_dataset.py train")
    st.stop()

uploaded = st.file_uploader(
    "Upload a road-traffic video",
    type=sorted(ALLOWED_EXTENSIONS),
    accept_multiple_files=False,
    help="Download the video first, then upload the local MP4/WEBM/MOV file. Videos must be no longer than 120 seconds.",
    max_upload_size=300,
)

if uploaded is not None:
    try:
        st.video(uploaded.getvalue())
        st.caption(f"Selected file: {uploaded.name} · {uploaded.size / (1024 * 1024):.1f} MB")
    except Exception:
        st.info("The browser preview may not support this video's codec. The analysis can still attempt to decode it.")

analyse = st.button(
    "Analyse video",
    type="primary",
    use_container_width=True,
    disabled=uploaded is None,
)

if analyse and uploaded is not None:
    try:
        input_path = _persist_upload(uploaded)
        metadata = probe(input_path, Config())
        with st.spinner("Extracting video frames and audio, then running the trained model..."):
            torch.set_num_threads(min(4, torch.get_num_threads()))
            prediction = predict_video(
                PROJECT_DIR,
                input_path,
                device_name=device_choice,
                visual_only=visual_only,
            )
        st.session_state["last_prediction"] = prediction
        st.session_state["last_video_name"] = uploaded.name
        st.session_state["last_video_metadata"] = metadata
    except Exception as exc:
        st.error(f"Analysis failed: {exc}")
        st.info(
            "Check that the model files exist, the virtual environment is active, "
            "the video is decodable, and its duration does not exceed 120 seconds."
        )

result = st.session_state.get("last_prediction")
if result:
    st.divider()
    st.subheader("Analysis result")
    predicted = str(result.get("prediction", "Unknown"))
    if predicted == "Accident":
        st.error("ACCIDENT DETECTED", icon="🚨")
    elif predicted == "Normal":
        st.success("NORMAL TRAFFIC", icon="✅")
    else:
        st.warning(f"Prediction: {predicted}")

    mode_title, mode_text = _mode_description(result)
    st.caption(f"Decision path: {mode_title}. {mode_text}")

    score = float(result.get("accident_score", 0.0))
    visual_score = float(result.get("visual_score", 0.0))
    audio_score = result.get("audio_score")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Final model score (0–1)", f"{score:.3f}")
    metric_cols[1].metric("Visual branch score", f"{visual_score:.3f}")
    metric_cols[2].metric("Audio branch score", "Unavailable" if audio_score is None else f"{float(audio_score):.3f}")
    metric_cols[3].metric("Video duration", f"{float(result.get('duration_s', 0)):.1f} s")
    st.caption("Scores are raw, uncalibrated model outputs—not verified probabilities or guarantees.")

    windows = result.get("windows", [])
    if windows:
        frame = pd.DataFrame(windows)
        if "start_s" in frame and "end_s" in frame:
            frame["window_midpoint_s"] = (frame["start_s"] + frame["end_s"]) / 2
            columns = {"visual_score": "Visual score"}
            if frame.get("audio_score") is not None and frame["audio_score"].notna().any():
                columns["audio_score"] = "Audio score"
            chart = frame.set_index("window_midpoint_s")[[key for key in columns if key in frame]].rename(columns=columns)
            st.subheader("Window-level model scores")
            st.line_chart(chart, y=list(chart.columns))
            display_cols = [c for c in ["start_s", "end_s", "visual_score", "audio_score", "audio_coverage"] if c in frame]
            st.dataframe(frame[display_cols], use_container_width=True, hide_index=True)

    warnings = result.get("warnings", [])
    if warnings:
        with st.expander("Model notes and limitations", expanded=True):
            for item in warnings:
                st.warning(str(item))

    json_data = json.dumps(result, indent=2, ensure_ascii=False).encode("utf-8")
    dl_cols = st.columns(2)
    dl_cols[0].download_button(
        "Download prediction JSON",
        data=json_data,
        file_name="accident_detection_result.json",
        mime="application/json",
        use_container_width=True,
    )
    dl_cols[1].download_button(
        "Download window report CSV",
        data=_csv_bytes(windows),
        file_name="accident_detection_windows.csv",
        mime="text/csv",
        use_container_width=True,
        disabled=not bool(windows),
    )

st.divider()
st.markdown(
    "<p class='muted'>Project: Audio-Visual Road Accident Detection · AIDL · "
    "Predictions are intended for academic demonstration and further evaluation.</p>",
    unsafe_allow_html=True,
)
