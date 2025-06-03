import streamlit as st
from PIL import Image
import numpy as np

st.title("Segmented Image Display Test")

# Create a simple test image
test_image = Image.new('RGB', (300, 300), color='blue')

# Create a simple segmented version (with black background)
segmented_image = Image.new('RGB', (300, 300), color='black')
# Add a white circle in the center
from PIL import ImageDraw
draw = ImageDraw.Draw(segmented_image)
draw.ellipse([100, 100, 200, 200], fill='white')

st.header("🔍 Test Segmentation Display")

st.subheader("Original vs Segmented Comparison")
col1, col2 = st.columns(2)

with col1:
    st.write("**Original**")
    st.image(test_image, use_column_width=True)

with col2:
    st.write("**Segmented**")
    st.image(segmented_image, use_column_width=True)

st.subheader("Segmented Image")
st.image(segmented_image, caption="This is how segmented images appear", use_column_width=True)

st.success("✅ If you can see the images above, segmented image display is working correctly!") 