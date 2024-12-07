import pandas as pd
import numpy as np
import streamlit as st
import easyocr
import PIL
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt

st.title("Get text from image with EasyOCR")
st.markdown("## EasyOCR with Streamlit")

# Upload image file
file = st.file_uploader(label="Upload your image", type=['png', 'jpg', 'jpeg'])
if file is not None:
    image = Image.open(file)  # Read image
    st.image(image)  # Display
    
    reader = easyocr.Reader(['tr', 'en'], gpu=False)
    result = reader.readtext(np.array(image))
    
    # Create dataframe
    textdic_easyocr = {}
    for idx in range(len(result)):
        pred_coor = result[idx][0]
        pred_text = result[idx][1]
        pred_confidence = result[idx][2]
        textdic_easyocr[pred_text] = {}
        textdic_easyocr[pred_text]['pred_confidence'] = pred_confidence

    df = pd.DataFrame.from_dict(textdic_easyocr).T
    st.table(df)

    # Add rectangles to image
    def rectangle(image, result):
        draw = ImageDraw.Draw(image)
        for res in result:
            top_left = tuple(res[0][0])
            bottom_right = tuple(res[0][2])
            draw.rectangle((top_left, bottom_right), outline="blue", width=2)
        st.image(image)

    rectangle(image, result)