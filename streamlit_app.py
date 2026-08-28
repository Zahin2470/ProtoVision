from __future__ import annotations

import io
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image

from protovision.pack import (
    PackFormatError,
    PackIncompatibleError,
    export_pack,
    import_pack,
)
from protovision.prototypes import PrototypeStore
from protovision.webdemo import embed_uploaded_image, get_backbone

# Page Configuration with wide viewport layout
st.set_page_config(
    page_title="ProtoVision AI — Neural Vision HUD",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Enterprise Cyber-SaaS Ultra-Premium Design System Injection
ULTRA_PREMIUM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    /* Global Typography Reset & CSS Variables */
    :root {
        --bg-void: #030712;
        --card-bg: rgba(15, 23, 42, 0.65);
        --card-border: rgba(255, 255, 255, 0.08);
        --accent-indigo: #6366F1;
        --accent-purple: #8B5CF6;
        --accent-cyan: #06B6D4;
        --accent-emerald: #10B981;
        --accent-rose: #F43F5E;
        --text-main: #F8FAFC;
        --text-muted: #94A3B8;
        --text-subtle: #64748B;
    }

    html, body, [class*="st-"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: var(--text-main);
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }
    
    /* Deep Space Ambient Glow Background */
    .stApp {
        background-color: var(--bg-void);
        background-image: 
            radial-gradient(circle at 15% 10%, rgba(99, 102, 241, 0.18) 0%, transparent 40%),
            radial-gradient(circle at 85% 15%, rgba(139, 92, 246, 0.15) 0%, transparent 45%),
            radial-gradient(circle at 50% 85%, rgba(6, 182, 212, 0.12) 0%, transparent 50%),
            radial-gradient(circle at 50% 30%, rgba(15, 23, 42, 0.5) 0%, transparent 100%);
        background-attachment: fixed;
    }

    /* Streamlit Chrome Elimination */
    header[data-testid="stHeader"] { visibility: hidden !important; height: 0px !important; }
    #MainMenu { visibility: hidden !important; }
    footer { visibility: hidden !important; }
    .block-container { 
        padding-top: 1.5rem !important; 
        padding-bottom: 2rem !important; 
        max-width: 1440px !important; 
    }

    /* LED Status Pulsing Badges */
    @keyframes pulse-emerald-glow {
        0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.6); }
        70% { box-shadow: 0 0 0 10px rgba(16, 185, 129, 0); }
        100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
    @keyframes pulse-amber-glow {
        0% { box-shadow: 0 0 0 0 rgba(245, 158, 11, 0.6); }
        70% { box-shadow: 0 0 0 10px rgba(245, 158, 11, 0); }
        100% { box-shadow: 0 0 0 0 rgba(245, 158, 11, 0); }
    }
    
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 10px;
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.12);
        padding: 7px 16px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        backdrop-filter: blur(16px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    .dot-emerald { width: 9px; height: 9px; background: #10B981; border-radius: 50%; animation: pulse-emerald-glow 2s infinite; }
    .dot-amber { width: 9px; height: 9px; background: #F59E0B; border-radius: 50%; animation: pulse-amber-glow 2s infinite; }

    /* Header Styling */
    .hud-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding-bottom: 1.5rem;
        margin-bottom: 1.8rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }
    .hud-title-container {
        display: flex;
        flex-direction: column;
        gap: 4px;
    }
    .hud-title {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.035em;
        background: linear-gradient(135deg, #FFFFFF 0%, #CBD5E1 50%, #818CF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .hud-badge {
        font-size: 0.75rem;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 6px;
        background: rgba(99, 102, 241, 0.2);
        border: 1px solid rgba(129, 140, 248, 0.4);
        color: #A5B4FC;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .hud-subtitle {
        color: var(--text-muted);
        font-size: 0.95rem;
        font-weight: 400;
        letter-spacing: -0.01em;
    }

    /* KPI Summary Card Matrix */
    .kpi-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.7) 0%, rgba(30, 41, 59, 0.4) 100%);
        border: 1px solid var(--card-border);
        border-radius: 16px;
        padding: 16px 22px;
        backdrop-filter: blur(20px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.36);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .kpi-card:hover {
        border-color: rgba(129, 140, 248, 0.3);
        transform: translateY(-2px);
    }
    .kpi-label {
        font-size: 0.73rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--text-subtle);
        font-weight: 700;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 1.55rem;
        font-weight: 700;
        color: var(--text-main);
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: -0.02em;
    }

    /* Tab Customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(15, 23, 42, 0.6);
        padding: 6px;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(16px);
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        border-radius: 10px;
        color: var(--text-muted);
        font-weight: 600;
        font-size: 0.9rem;
        border: none !important;
        padding: 0 22px;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.3) 0%, rgba(139, 92, 246, 0.3) 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(129, 140, 248, 0.5) !important;
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.25);
    }

    /* Styled Container Boxes */
    div[data-testid="stForm"], div[data-testid="stContainer"] {
        background: var(--card-bg) !important;
        border: 1px solid var(--card-border) !important;
        border-radius: 18px !important;
        backdrop-filter: blur(20px) !important;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.25) !important;
        padding: 1.25rem !important;
    }

    /* Buttons Override */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.9rem;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        border: 1px solid rgba(255, 255, 255, 0.14);
        background: rgba(30, 41, 59, 0.7);
        color: var(--text-main);
        padding: 0.5rem 1rem;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%) !important;
        border: 1px solid rgba(165, 180, 252, 0.4) !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 20px rgba(79, 70, 229, 0.4);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.3);
        border-color: rgba(129, 140, 248, 0.6);
    }

    /* File Uploader Container */
    div[data-testid="stFileUploader"] {
        background: rgba(15, 23, 42, 0.4);
        border: 1px dashed rgba(255, 255, 255, 0.18);
        border-radius: 14px;
        padding: 12px;
        transition: border-color 0.2s ease;
    }
    div[data-testid="stFileUploader"]:hover {
        border-color: var(--accent-indigo);
    }

    /* Inputs */
    .stTextInput input {
        background-color: rgba(15, 23, 42, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 10px !important;
        color: var(--text-main) !important;
        padding: 10px 14px !important;
    }
    .stTextInput input:focus {
        border-color: var(--accent-indigo) !important;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.3) !important;
    }
    
    /* Result HUD Cards */
    .match-card-success {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.18) 0%, rgba(5, 150, 105, 0.05) 100%);
        border: 1px solid rgba(16, 185, 129, 0.5);
        border-radius: 14px;
        padding: 18px;
        margin-bottom: 1.2rem;
        box-shadow: 0 8px 24px rgba(16, 185, 129, 0.15);
    }
    .match-card-failed {
        background: linear-gradient(135deg, rgba(244, 63, 94, 0.18) 0%, rgba(190, 18, 60, 0.05) 100%);
        border: 1px solid rgba(244, 63, 94, 0.5);
        border-radius: 14px;
        padding: 18px;
        margin-bottom: 1.2rem;
        box-shadow: 0 8px 24px rgba(244, 63, 94, 0.15);
    }
</style>
"""
st.markdown(ULTRA_PREMIUM_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Initializing Neural Backbone Architecture...")
def _cached_backbone():
    return get_backbone()


def _get_store() -> PrototypeStore:
    if "store" not in st.session_state:
        st.session_state.store = PrototypeStore()
    return st.session_state.store


def _open_upload(uploaded_file) -> Image.Image:
    return Image.open(io.BytesIO(uploaded_file.getvalue()))


def main():
    backbone, is_real = _cached_backbone()
    store = _get_store()

    # --- Header Bar ---
    status_html = (
        '<div class="status-pill"><span class="dot-emerald"></span> DINOv3 Neural Backbone Active</div>'
        if is_real
        else '<div class="status-pill"><span class="dot-amber"></span> Neural Fallback Mode</div>'
    )
    st.markdown(
        f"""
        <div class="hud-header">
            <div class="hud-title-container">
                <h1 class="hud-title">🔎 ProtoVision </h1>
                <div class="hud-subtitle">Self-supervised few-shot visual target recognition & continuous feature indexing</div>
            </div>
            <div>{status_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- KPI HUD Matrix ---
    total_classes = len(store.labels())
    total_examples = sum(store.example_count(lbl) for lbl in store.labels())

    col_k1, col_k2, col_k3 = st.columns(3)
    col_k1.markdown(
        f"""<div class="kpi-card">
            <div class="kpi-label">Vision Extraction Model</div>
            <div class="kpi-value">{"DINOv3-Enterprise" if is_real else "Synthetic Fallback"}</div>
        </div>""",
        unsafe_allow_html=True,
    )
    col_k2.markdown(
        f"""<div class="kpi-card">
            <div class="kpi-label">Indexed Target Classes</div>
            <div class="kpi-value">{total_classes}</div>
        </div>""",
        unsafe_allow_html=True,
    )
    col_k3.markdown(
        f"""<div class="kpi-card">
            <div class="kpi-label">Registered Feature Vectors</div>
            <div class="kpi-value">{total_examples}</div>
        </div>""",
        unsafe_allow_html=True,
    )

    st.write("")

    if not is_real:
        st.warning(
            "System running in **Synthetic Fallback Mode**. Functional interactions remain available, "
            "but true vision extraction requires loading official DINOv3 model weights (`docs/DINOV3_SETUP.md`).",
            icon="⚡",
        )

    # --- Navigation Workspace ---
    tab_enroll, tab_recognize, tab_pack = st.tabs(
        ["➕ Enroll Target Class", "🎯 Visual Recognition HUD", "📦 Model Pack Manager"]
    )

    # -- Tab 1: Enroll Class --------------------------------------------
    with tab_enroll:
        col_left, col_right = st.columns([1, 1], gap="large")

        with col_left:
            with st.container(border=True):
                st.markdown("### ➕ Register New Target Vector")
                st.caption("Provide visual image samples from multiple angles to generate normalized prototype embeddings.")

                label = st.text_input("Class Identifier", placeholder="e.g. keycard, optical_sensor, headset")
                uploads = st.file_uploader(
                    "Upload Reference Images (3–10 samples recommended)",
                    type=["png", "jpg", "jpeg"],
                    accept_multiple_files=True,
                )

                if uploads:
                    st.caption("Sample Image Previews")
                    preview_cols = st.columns(min(len(uploads), 4))
                    for idx, upload in enumerate(uploads):
                        preview_cols[idx % 4].image(upload, use_container_width=True)

                can_add = bool(label.strip()) and bool(uploads)
                if st.button("✨ Enroll Class Vectors", type="primary", disabled=not can_add, use_container_width=True):
                    for upload in uploads:
                        image = _open_upload(upload)
                        embedding = embed_uploaded_image(backbone, image)
                        store.add_example(label.strip(), embedding)
                    st.toast(f"Successfully registered '{label.strip()}' with {len(uploads)} vector sample(s).", icon="✅")
                    st.rerun()

        with col_right:
            with st.container(border=True):
                st.markdown("### 🧠 Active Prototype Registry")
                st.caption("Prototype feature vectors held in active memory available for real-time inference.")

                if store.is_empty():
                    st.info("No targets registered. Use the panel on the left to enroll your first class.")
                else:
                    for enrolled_label in sorted(store.labels()):
                        count = store.example_count(enrolled_label)
                        c1, c2 = st.columns([3, 1])
                        c1.markdown(
                            f"**{enrolled_label}**  \n"
                            f"<span style='color:#94A3B8; font-size:0.82rem;'>{count} prototype sample vector(s)</span>", 
                            unsafe_allow_html=True
                        )
                        if c2.button("🗑️ Remove", key=f"remove_{enrolled_label}", use_container_width=True):
                            store.remove_class(enrolled_label)
                            st.rerun()

    # -- Tab 2: Visual Recognition HUD ----------------------------------
    with tab_recognize:
        if store.is_empty():
            st.info("Enroll at least one target class in the **Enroll Target Class** tab to activate visual inference.")
        else:
            col_in, col_out = st.columns([1, 1], gap="large")

            with col_in:
                with st.container(border=True):
                    st.markdown("### 🎯 Visual Query Input")
                    query_upload = st.file_uploader(
                        "Upload unknown visual target image", type=["png", "jpg", "jpeg"], key="query_upload"
                    )
                    threshold = st.slider("Match Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

                    if query_upload:
                        st.image(query_upload, caption="Target Query Visual Probe", use_container_width=True)

            with col_out:
                with st.container(border=True):
                    st.markdown("### 📊 Inference HUD Telemetry")
                    if not query_upload:
                        st.caption("Awaiting visual query upload...")
                    else:
                        image = _open_upload(query_upload)
                        embedding = embed_uploaded_image(backbone, image)

                        result = store.best_match(embedding, threshold=threshold)
                        similarities = store.all_similarities(embedding)

                        if result.is_known:
                            st.markdown(
                                f"""
                                <div class="match-card-success">
                                    <div style="font-size:0.75rem; color:#34D399; font-weight:700; text-transform:uppercase; letter-spacing:0.05em;">Visual Match Confirmed</div>
                                    <div style="font-size:1.5rem; font-weight:800; color:#FFFFFF; margin-top:2px;">{result.label}</div>
                                    <div style="font-family:'JetBrains Mono', monospace; font-size:0.92rem; color:#A7F3D0; margin-top:6px;">Cosine Similarity: {result.similarity:.4f}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                f"""
                                <div class="match-card-failed">
                                    <div style="font-size:0.75rem; color:#F87171; font-weight:700; text-transform:uppercase; letter-spacing:0.05em;">Target Unidentified</div>
                                    <div style="font-size:1.4rem; font-weight:800; color:#FFFFFF; margin-top:2px;">Unknown Class</div>
                                    <div style="font-family:'JetBrains Mono', monospace; font-size:0.88rem; color:#FECACA; margin-top:6px;">Best Candidate: {result.label if result.label else "None"} ({result.similarity:.4f}) — Below threshold {threshold:.2f}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        st.markdown("<div style='font-size:0.9rem; font-weight:700; margin-bottom:14px;'>Similarity Distribution Spectrum</div>", unsafe_allow_html=True)
                        sorted_sims = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
                        
                        # Render HTML progress bars for visual telemetry
                        for class_name, sim_score in sorted_sims:
                            percentage = max(0, min(int(sim_score * 100), 100))
                            st.markdown(
                                f"""
                                <div style="margin-bottom: 14px;">
                                    <div style="display:flex; justify-content:space-between; align-items:center; font-size: 0.85rem; font-weight: 600; color:#E2E8F0; margin-bottom: 5px;">
                                        <span>{class_name}</span>
                                        <span style="font-family:'JetBrains Mono', monospace; color:#818CF8; font-weight:700;">{sim_score:.4f}</span>
                                    </div>
                                    <div style="width: 100%; height: 9px; background: rgba(255,255,255,0.06); border-radius: 6px; overflow: hidden; border: 1px solid rgba(255,255,255,0.05);">
                                        <div style="width: {percentage}%; height: 100%; background: linear-gradient(90deg, #6366F1 0%, #A855F7 50%, #06B6D4 100%); border-radius: 6px; box-shadow: 0 0 10px rgba(99, 102, 241, 0.5);"></div>
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

    # -- Tab 3: Model Pack Manager --------------------------------------
    with tab_pack:
        col_exp, col_imp = st.columns([1, 1], gap="large")

        with col_exp:
            with st.container(border=True):
                st.markdown("### 💾 Export Recognizer Pack")
                st.caption("Serialize registered feature prototype vectors into a portable `.json` deployment pack file.")

                if store.is_empty():
                    st.info("Registry is empty. Register visual target classes to generate export packs.")
                else:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        pack_path = Path(tmpdir) / "recognizer_pack.json"
                        export_pack(store, pack_path)
                        st.download_button(
                            "💾 Export Serialized Pack (.json)",
                            data=pack_path.read_bytes(),
                            file_name="recognizer_pack.json",
                            mime="application/json",
                            use_container_width=True,
                            type="primary",
                        )

        with col_imp:
            with st.container(border=True):
                st.markdown("### 📥 Import Recognizer Pack")
                st.caption("Load visual vector stores from external JSON recognizer packages.")

                pack_upload = st.file_uploader("Select `.json` Model Pack", type=["json"], key="pack_upload")
                on_conflict = st.radio(
                    "Conflict Resolution Policy",
                    options=["skip", "merge", "overwrite"],
                    format_func=lambda x: x.capitalize(),
                    horizontal=True,
                )

                if pack_upload and st.button("📤 Ingest Model Pack", use_container_width=True):
                    with tempfile.TemporaryDirectory() as tmpdir:
                        tmp_path = Path(tmpdir) / "uploaded_pack.json"
                        tmp_path.write_bytes(pack_upload.getvalue())
                        try:
                            summary = import_pack(store, tmp_path, on_conflict=on_conflict)
                        except (PackFormatError, PackIncompatibleError) as exc:
                            st.error(str(exc))
                        else:
                            if summary.added:
                                st.success(f"Added Classes: {', '.join(sorted(summary.added))}")
                            if summary.merged:
                                st.success(f"Merged Classes: {', '.join(sorted(summary.merged))}")
                            if summary.overwritten:
                                st.success(f"Overwritten Classes: {', '.join(sorted(summary.overwritten))}")
                            if summary.skipped:
                                st.warning(f"Skipped Classes: {', '.join(sorted(summary.skipped))}")
                            for warning in summary.warnings:
                                st.warning(warning)
                            if summary.changed_any_class:
                                st.rerun()

    st.divider()
    st.markdown(
        "<div style='text-align:center; font-size:0.82rem; color:#64748B;'>⚡ ProtoVision Core Engine · Web Neural HUD Edition · Launch local executable (<code>main.py live</code>) for hardware video stream overlays</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()