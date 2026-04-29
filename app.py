# app.py
# Streamlit UI — elderly friendly prescription assistant

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import pytesseract
from dotenv import load_dotenv

load_dotenv()

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Import your existing modules
from pipeline      import (preprocess_for_prescription,
                            parse_medicines,
                            ocr_packaging,
                            verify_medicine,
                            get_confidence_level)
from rag_engine    import explain_prescription


# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────

st.set_page_config(
    page_title = "Medicine Helper",
    page_icon  = "💊",
    layout     = "wide"
)

st.markdown("""
<style>
    .big-title {
        font-size: 2.8rem;
        font-weight: bold;
        color: #1a5276;
        text-align: center;
        padding: 20px 0;
    }
    .section-title {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2874a6;
        margin: 20px 0 10px 0;
    }
    .medicine-card {
        background-color : #eaf4fb;
        border-left      : 6px solid #2874a6;
        padding          : 15px 20px;
        border-radius    : 8px;
        margin           : 10px 0;
        font-size        : 1.1rem;
        line-height      : 1.8;
    }
    .correct-box {
        background-color : #d5f5e3;
        border-left      : 6px solid #27ae60;
        padding          : 20px;
        border-radius    : 8px;
        font-size        : 1.2rem;
        line-height      : 1.8;
    }
    .warning-box {
        background-color : #fef9e7;
        border-left      : 6px solid #f39c12;
        padding          : 20px;
        border-radius    : 8px;
        font-size        : 1.2rem;
        line-height      : 1.8;
    }
    .wrong-box {
        background-color : #fadbd8;
        border-left      : 6px solid #e74c3c;
        padding          : 20px;
        border-radius    : 8px;
        font-size        : 1.2rem;
        line-height      : 1.8;
    }
    .explanation-box {
        font-size   : 1.15rem;
        line-height : 2.0;
        color       : #1a1a1a;
        padding     : 10px 0;
    }
    .stButton > button {
        font-size     : 1.1rem !important;
        padding       : 14px 28px !important;
        border-radius : 8px !important;
        width         : 100%;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# SESSION STATE INIT
# ──────────────────────────────────────────────

def init_state():
    defaults = {
        'medicines'         : [],
        'prescription_done' : False,
        'ocr_raw_text'      : '',
        'explanation'       : '',
        'prefill_question'  : '',
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_state()


# ──────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────

def decode_dosage(dosage_str):
    """Convert 1-0-1 to readable Morning and Night."""
    if dosage_str == "Not found":
        return "As directed by doctor"
    parts = dosage_str.split('-')
    if len(parts) != 3:
        return dosage_str
    labels = ['Morning', 'Afternoon', 'Night']
    times  = [labels[i] for i, p in enumerate(parts) if p.strip() == '1']
    return ' + '.join(times) if times else "As directed"


def run_prescription_ocr(uploaded_file):
    """Read uploaded prescription image and extract medicines."""
    bytes_data = uploaded_file.read()
    nparr      = np.frombuffer(bytes_data, np.uint8)
    img        = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    preprocessed = preprocess_for_prescription(img)
    pil_img      = Image.fromarray(preprocessed)
    text         = pytesseract.image_to_string(pil_img, config='--psm 6')
    return text, img


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────

st.sidebar.markdown("## 💊 Medicine Helper")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠  Home",
        "📄  Step 1 — Upload Prescription",
        "💬  Step 2 — Understand Your Medicines",
        "📦  Step 3 — Check Medicine Box",
        "❓  Step 4 — Ask a Question",
    ],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### How to use")
st.sidebar.markdown("""
**Step 1** — Upload your prescription photo

**Step 2** — Get a simple explanation

**Step 3** — Check your medicine box

**Step 4** — Ask any question
""")

if st.session_state.prescription_done:
    st.sidebar.markdown("---")
    st.sidebar.success(
        f"✅ Prescription loaded\n\n"
        f"{len(st.session_state.medicines)} medicines found"
    )


# ──────────────────────────────────────────────
# PAGE: HOME
# ──────────────────────────────────────────────

if page == "🏠  Home":
    st.markdown(
        '<div class="big-title">💊 Medicine Helper</div>',
        unsafe_allow_html=True
    )
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("### 📄 Step 1")
        st.markdown("Upload a photo of your **prescription**")

    with col2:
        st.markdown("### 💬 Step 2")
        st.markdown("Get a **simple explanation** of your medicines")

    with col3:
        st.markdown("### 📦 Step 3")
        st.markdown("**Check your medicine box** is correct")

    with col4:
        st.markdown("### ❓ Step 4")
        st.markdown("**Ask any question** about your medicines")

    st.markdown("---")
    st.info(
        "👆 Use the menu on the **left side** to go through each step. "
        "Start with **Step 1**."
    )


# ──────────────────────────────────────────────
# PAGE: STEP 1 — UPLOAD PRESCRIPTION
# ──────────────────────────────────────────────

elif page == "📄  Step 1 — Upload Prescription":
    st.markdown(
        '<div class="section-title">📄 Upload Your Prescription</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        "Take a **clear photo** of your prescription paper "
        "and upload it here."
    )

    uploaded = st.file_uploader(
        "Choose your prescription image",
        type=['jpg', 'jpeg', 'png']
    )

    if uploaded:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("**Your prescription:**")
            st.image(uploaded, use_column_width=True)

        with col2:
            with st.spinner("🔍 Reading your prescription..."):
                raw_text, _ = run_prescription_ocr(uploaded)
                medicines   = parse_medicines(raw_text)

                st.session_state.medicines         = medicines
                st.session_state.ocr_raw_text      = raw_text
                st.session_state.prescription_done = True

            if medicines:
                st.success(
                    f"✅ Found **{len(medicines)} medicines** "
                    f"in your prescription"
                )
                st.markdown("---")
                st.markdown("**Your medicines:**")

                for i, med in enumerate(medicines, 1):
                    dose = decode_dosage(med['dosage'])
                    st.markdown(
                        f'<div class="medicine-card">'
                        f'<b>{i}. {med["name"]}</b><br>'
                        f'⏰ <b>Take:</b> {dose}<br>'
                        f'📋 <b>Schedule:</b> {med["timing"]}'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                st.markdown("---")
                st.info("✅ Now go to **Step 2** for a simple explanation.")

            else:
                st.error(
                    "❌ Could not read medicines from this image.\n\n"
                    "Please try a clearer, better-lit photo."
                )

        with st.expander("🔬 Show raw OCR text (for troubleshooting)"):
            st.text(raw_text)


# ──────────────────────────────────────────────
# PAGE: STEP 2 — UNDERSTAND MEDICINES
# ──────────────────────────────────────────────

elif page == "💬  Step 2 — Understand Your Medicines":
    st.markdown(
        '<div class="section-title">💬 Understanding Your Medicines</div>',
        unsafe_allow_html=True
    )

    if not st.session_state.prescription_done:
        st.warning("⚠️ Please complete Step 1 first.")
        st.stop()

    medicines = st.session_state.medicines

    # Medicine schedule table
    st.markdown("### 📅 Your Daily Medicine Schedule")

    table_data = []
    for med in medicines:
        parts = (
            med['dosage'].split('-')
            if med['dosage'] != 'Not found'
            else ['?', '?', '?']
        )
        table_data.append({
            'Medicine'  : med['name'],
            'Morning'   : '✅' if len(parts) > 0
                          and parts[0].strip() == '1' else '—',
            'Afternoon' : '✅' if len(parts) > 1
                          and parts[1].strip() == '1' else '—',
            'Night'     : '✅' if len(parts) > 2
                          and parts[2].strip() == '1' else '—',
            'Duration'  : med['timing'],
        })

    st.table(table_data)
    st.markdown("---")

    # RAG explanation
    st.markdown("### 🤖 Simple Explanation")
    st.markdown(
        "Click the button below to get a simple explanation "
        "of all your medicines."
    )

    if st.button("🔊 Explain My Medicines in Simple Words"):
        with st.spinner(
            "🤖 Generating explanation — usually takes 3 to 5 seconds..."
        ):
            explanation = explain_prescription(medicines)
            st.session_state.explanation = explanation

    if st.session_state.explanation:
        st.markdown("---")
        st.markdown(
            f'<div class="explanation-box">'
            f'{st.session_state.explanation.replace(chr(10), "<br>")}'
            f'</div>',
            unsafe_allow_html=True
        )
        st.markdown("---")
        st.download_button(
            label    = "📥 Download this explanation",
            data     = st.session_state.explanation,
            file_name= "my_prescription_explanation.txt",
            mime     = "text/plain"
        )


# ──────────────────────────────────────────────
# PAGE: STEP 3 — CHECK MEDICINE BOX
# ──────────────────────────────────────────────

elif page == "📦  Step 3 — Check Medicine Box":
    st.markdown(
        '<div class="section-title">📦 Check Your Medicine Box</div>',
        unsafe_allow_html=True
    )

    if not st.session_state.prescription_done:
        st.warning("⚠️ Please complete Step 1 first.")
        st.stop()

    st.markdown(
        "Upload a photo of the **medicine box or strip** "
        "you are about to take. "
        "We will check if it matches your prescription."
    )

    packaging_file = st.file_uploader(
        "Choose medicine packaging image",
        type=['jpg', 'jpeg', 'png'],
        key="packaging"
    )

    if packaging_file:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("**Your medicine packaging:**")
            st.image(packaging_file, use_column_width=True)

        with col2:
            with st.spinner("🔍 Reading medicine name from packaging..."):
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix='.jpg'
                ) as tmp:
                    tmp.write(packaging_file.read())
                    tmp_path = tmp.name

                detected = ocr_packaging(tmp_path)
                os.unlink(tmp_path)

            if detected:
                st.markdown(f"**Name detected on box:** `{detected}`")
                st.markdown("---")

                matched, score = verify_medicine(
                    detected,
                    st.session_state.medicines
                )
                confidence = get_confidence_level(score)

                if matched and score >= 85:
                    dose = decode_dosage(matched['dosage'])
                    st.markdown(
                        f'<div class="correct-box">'
                        f'✅ <b>CORRECT MEDICINE</b><br><br>'
                        f'This medicine is in your prescription.<br><br>'
                        f'💊 <b>Medicine :</b> {matched["name"]}<br>'
                        f'⏰ <b>Take it  :</b> {dose}<br>'
                        f'📋 <b>Schedule :</b> {matched["timing"]}<br>'
                        f'📊 <b>Confidence:</b> {confidence} ({score}/100)'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                elif matched and score >= 75:
                    st.markdown(
                        f'<div class="warning-box">'
                        f'⚠️ <b>PLEASE DOUBLE CHECK</b><br><br>'
                        f'This looks like it could be: '
                        f'<b>{matched["name"]}</b><br>'
                        f'But we are only moderately sure '
                        f'({score}/100).<br><br>'
                        f'Please show this box to your pharmacist '
                        f'to confirm before taking it.'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                else:
                    st.markdown(
                        f'<div class="wrong-box">'
                        f'❌ <b>WARNING — WRONG MEDICINE</b><br><br>'
                        f'<b>{detected}</b> is NOT in your prescription.<br><br>'
                        f'Please do <b>NOT</b> take this medicine.<br>'
                        f'Contact your doctor or pharmacist immediately.'
                        f'</div>',
                        unsafe_allow_html=True
                    )
            else:
                st.error(
                    "❌ Could not read the medicine name from this image.\n\n"
                    "Please try a clearer photo with better lighting."
                )


# ──────────────────────────────────────────────
# PAGE: STEP 4 — ASK A QUESTION
# ──────────────────────────────────────────────

elif page == "❓  Step 4 — Ask a Question":
    st.markdown(
        '<div class="section-title">❓ Ask About Your Medicines</div>',
        unsafe_allow_html=True
    )

    if not st.session_state.prescription_done:
        st.warning("⚠️ Please complete Step 1 first.")
        st.stop()

    st.markdown(
        "You can ask anything about your medicines. "
        "Write in simple words — just like asking a friend or family member."
    )

    # Suggested questions
    st.markdown("### 💡 Common Questions")
    st.markdown("Click any question below or type your own:")

    suggested = [
        "Can I take all my medicines at the same time?",
        "What happens if I miss a dose?",
        "Should I take these medicines with food?",
        "What side effects should I watch out for?",
        "Can I stop taking the medicines if I feel better?",
        "Are there any foods I should avoid?",
    ]

    cols = st.columns(2)
    for i, question in enumerate(suggested):
        with cols[i % 2]:
            if st.button(question, key=f"q_{i}"):
                st.session_state.prefill_question = question

    st.markdown("---")

    user_question = st.text_area(
        "Your question:",
        value   = st.session_state.prefill_question,
        height  = 100,
        placeholder = "Type your question here..."
    )

    if st.button("🔍 Get Answer"):
        if user_question.strip():
            with st.spinner("🤖 Finding answer — usually 3 to 5 seconds..."):
                answer = explain_prescription(
                    st.session_state.medicines,
                    user_question=user_question
                )
            st.markdown("---")
            st.markdown("### 💬 Answer:")
            st.markdown(
                f'<div class="explanation-box">'
                f'{answer.replace(chr(10), "<br>")}'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            st.warning("Please type or select a question first.")