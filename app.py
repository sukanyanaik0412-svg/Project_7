import streamlit as st
import base64
import joblib
import re
from pathlib import Path

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Set page configuration
st.set_page_config(
    page_title="Disaster Tweet Classifier",
    layout="centered",  # "wide" or "centered"
)

# Set background image using base64
def add_bg_image(image_path):
    image_file = Path(__file__).resolve().parent / image_path
    if not image_file.is_file():
        return

    encoded = base64.b64encode(image_file.read_bytes()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-attachment: fixed;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Add your background image (replace 'background.jpg' with your file)
add_bg_image("background.jpg")

# Custom CSS styles
st.markdown("""
    <style>
        .stApp {
            background-color: #eef2ed;
        }
        .main {
            font-family: 'Segoe UI', sans-serif;
        }
        .title-box {
            background-color: rgba(29, 50, 44, 0.92);
            color: #ffffff;
            padding: 24px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.3);
        }
        .title-box h1 {
            font-size: 38px;
            line-height: 1.15;
            margin: 0 0 8px;
        }
        .title-box p {
            font-size: 16px;
            color: #dce7df;
            margin: 0;
        }
        .stTextArea textarea {
            font-size: 16px;
            padding: 12px;
            border-radius: 6px;
        }
        .stTextArea label {
            color: #26352f;
            font-weight: 600;
        }
        .stButton > button {
            background-color: #c4443d;
            color: white;
            font-weight: bold;
            border: 0;
            border-radius: 6px;
            padding: 10px 20px;
            font-size: 16px;
            transition: background-color 150ms ease, transform 150ms ease;
        }
        .stButton > button:hover {
            background-color: #a73530;
            color: white;
            transform: translateY(-1px);
        }
        .stButton > button:focus-visible {
            outline: 3px solid #78a08e;
            outline-offset: 2px;
        }
        .result-box {
            background-color: rgba(255, 255, 255, 0.92);
            padding: 18px;
            border-left: 4px solid #c4443d;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Title box
st.markdown("""
    <div class="title-box">
        <h1> Disaster Tweet Classifier</h1>
        <p>Instantly detect if a tweet is about a real disaster or not</p>
    </div>
""", unsafe_allow_html=True)

# Tweet input section
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    tweet = st.text_area("✍️ Paste your tweet below:", height=180)

    if st.button("🚀 Classify Tweet"):
        if not tweet.strip():
            st.warning("⚠️ Please enter a tweet to classify.")
        else:
            # Text cleaning
            def clean_text(text):
                text = text.lower()
                text = re.sub(r"http\S+|www\S+|https\S+", '', text)
                text = re.sub(r"@\w+|#\w+", '', text)
                text = re.sub(r"[^\w\s]", '', text)
                return text.strip()

            cleaned = clean_text(tweet)
            vector = vectorizer.transform([cleaned])
            pred = model.predict(vector)[0]
            proba = model.predict_proba(vector)[0][pred] if hasattr(model, "predict_proba") else None

            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            st.markdown(
    "<span style='color:black; font-size:20px; font-weight:bold'>🔎 Prediction Result:</span>",
    unsafe_allow_html=True
)

            if pred == 1:
                st.error(f"🚨 **Disaster Tweet Detected!**\n\nConfidence: `{proba:.2%}`" if proba else "🚨 **Disaster Tweet Detected!**")
            else:
                st.success(f"✅ **Not a Disaster Tweet**\n\nConfidence: `{proba:.2%}`" if proba else "✅ **Not a Disaster Tweet**")
            st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    "<center><span style='color:black'>🌍 Built using Streamlit | NLP & Machine Learning</span></center>",
    unsafe_allow_html=True
)