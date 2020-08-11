"""Streamlit web app"""


import streamlit as st
from PIL import Image

import numpy as np

from retinaface.pre_trained_models import get_model
from retinaface.utils import vis_annotations

st.set_option("deprecation.showfileUploaderEncoding", False)

model = get_model("resnet50_2020-07-20", max_size=1024, device="cpu")
model.eval()

st.title("Detect faces and key points.")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = np.array(Image.open(uploaded_file))
    st.image(image, caption="Before", use_column_width=True)
    st.write("")
    st.write("Detecting faces...")
    annotations = model.predict_jsons(image)

    visualized_image = vis_annotations(image, annotations)

    st.image(visualized_image, caption="After", use_column_width=True)
