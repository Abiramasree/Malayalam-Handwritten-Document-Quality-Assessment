
import streamlit as st
import numpy as np
import cv2
import time

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

st.set_page_config(
    page_title="Malayalam Handwritten Document Quality Assessment",
    page_icon="📄",
    layout="wide"
)

st.markdown("""
<style>
.main{background-color:#F5F7FB;}
.block-container{padding-top:2rem;}
.hero{
background:linear-gradient(90deg,#2563EB,#06B6D4);
padding:30px;border-radius:18px;text-align:center;color:white;
margin-bottom:25px;
}
.hero h1{margin:0;font-size:38px;}
.hero p{margin-top:10px;font-size:18px;}
.footer{text-align:center;color:gray;margin-top:30px;}
</style>
""", unsafe_allow_html=True)

model = load_model("document_quality_model.keras")
class_names = ["Average","Good","Poor"]

st.markdown("""
<div class="hero">
<h1>📄 Malayalam Handwritten Document Quality Assessment</h1>
<p>Upload a handwritten document image and instantly assess its quality using AI.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("## 📂 Upload Document")

uploaded_file = st.file_uploader(
    "Choose a handwritten document image",
    type=["png","jpg","jpeg"]
)

if uploaded_file is not None:

    with st.spinner("🔍 Analyzing document..."):
        time.sleep(1)

        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    col1, col2 = st.columns([1.2,1])

    with col1:
        st.markdown("## 🖼 Uploaded Image")
        st.image(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            use_container_width=True
        )

    img = cv2.resize(image,(224,224))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = preprocess_input(img)
    img = np.expand_dims(img,axis=0)

    prediction = model.predict(img, verbose=0)

    index = np.argmax(prediction)
    confidence = float(np.max(prediction)*100)

    with col2:

        st.markdown("## 🤖 Assessment Result")

        if class_names[index] == "Good":
            st.success("✅ Quality : Good")

        elif class_names[index] == "Average":
            st.warning("🟡 Quality : Average")

        else:
            st.error("🔴 Quality : Poor")

        st.metric("Confidence", f"{confidence:.2f}%")
        st.progress(confidence/100)

        st.markdown("---")
        st.markdown("### 📊 Prediction Probability")

        for i, name in enumerate(class_names):

            probability = float(prediction[0][i] * 100)

            st.write(f"**{name}**")
            st.progress(probability/100)
            st.caption(f"Probability: {probability:.2f}%")

st.markdown("---")
st.markdown("""
<div class="footer">
© 2026 Malayalam Handwritten Document Quality Assessment System
</div>
""", unsafe_allow_html=True)
