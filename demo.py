import streamlit as st
from PIL import Image
import cv2
import numpy as np
from google import genai
import asyncio
import logging
import tempfile
import os
from segmentation_module import GeminiSegmentationModel

from utils import apply_mask, calculate_ciede2000_color_similarity, create_generic_mask

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Image Processing and Feature Extraction ---


def resize_image_to_fixed_size(pil_image):
    """Resizes an image to a fixed size of (384, 512) for all processing."""
    fixed_size = (384, 512)
    if pil_image.size == fixed_size:
        return pil_image
    resized_image = pil_image.resize(fixed_size, Image.Resampling.LANCZOS)
    print(f"Resized image from {pil_image.size} to {fixed_size} for processing")
    return resized_image


def extract_features(pil_image, mask, image_name=""):
    """Extracts dominant color, texture histogram, and APL from a PIL image, optionally using a mask."""
    if pil_image is None:
        st.warning(f"Cannot extract features for {image_name}: Input image is None.")
        return None
    try:
        # Ensure image is RGB for feature extraction
        image_rgb = pil_image.convert("RGB")
        img_np = np.array(image_rgb)

        mask_np = None
        if mask is not None:
            # Ensure mask is L and the same size as the image
            if mask.size != pil_image.size:
                mask = mask.resize(pil_image.size, Image.Resampling.LANCZOS)
            if mask.mode != "L":
                mask = mask.convert("L")
            mask_np = np.array(mask)

        # Dominant Color
        if mask_np is not None:
            pixels = np.float32(img_np[mask_np > 0])
            if pixels.size == 0:
                st.warning(
                    f"No pixels in mask for {image_name}. Cannot extract features."
                )
                return None
        else:
            pixels = np.float32(img_np.reshape(-1, 3))

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, _, centers = cv2.kmeans(
            pixels, 1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )
        dominant_color = tuple(np.uint8(centers[0]))

        # Average Pixel Luminance (APL)
        gray_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        if mask_np is not None and np.any(mask_np > 0):
            apl = np.mean(gray_img[mask_np > 0])
        else:
            apl = np.mean(gray_img)

        return {
            "dominant_color": dominant_color,
            "apl": apl,
        }
    except Exception as e:
        st.warning(f"Could not extract features for {image_name}: {e}")
        return None


# --- Gemini API and Segmentation ---


async def identify_garment_characteristics(
    pil_image, api_key, model_name="gemini-2.5-flash-preview-05-20", max_retries=2
):
    """Function to identify all garments in the reference image."""
    if pil_image is None or not api_key:
        return "Invalid input"

    client = genai.Client(api_key=api_key)

    for attempt in range(max_retries):
        try:
            prompt = """Identify ALL garments visible in this image. For each garment, provide a brief description including:
- Garment type and main color
- Basic style/fit

Keep descriptions short and concise. Format as a numbered list. Examples:
1. Blue fitted t-shirt
2. Black straight-leg jeans  
3. White casual sneakers

Focus on the most visible and prominent garments."""

            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model=model_name, contents=[prompt, pil_image]
                ),
            )
            return response.text.strip()
        except Exception as e:
            print(f"Error during garment identification (attempt {attempt + 1}): {e}")
            if attempt >= max_retries - 1:
                return f"Failed to identify garments after {max_retries} attempts."
            await asyncio.sleep(2)
    return "Unknown Garments"


async def analyze_pattern_descriptions(
    pil_image,
    image_name,
    api_key,
    model_name="gemini-2.5-flash-preview-05-20",
    max_retries=2,
):
    """Function to analyze and describe all patterns visible in an image."""
    if pil_image is None or not api_key:
        return f"Invalid input for {image_name}"

    client = genai.Client(api_key=api_key)

    for attempt in range(max_retries):
        try:
            prompt = """Analyze this garment image and list the patterns visible. For each pattern, provide only these four attributes:

* Pattern type: (e.g., stripes, florals, geometric, solid, plaid, dots, checks, etc.)
* Pattern colors: (list the main colors used in the pattern)
* Orientation: (horizontal, vertical, diagonal, random, radial, etc.)
* Spacing: (tight, medium, wide, irregular, etc.)

Format as a simple bulleted list. If multiple distinct patterns exist, list each separately. If it's a solid color with no pattern, state "Solid color - no pattern"."""

            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model=model_name, contents=[prompt, pil_image]
                ),
            )
            return response.text.strip()
        except Exception as e:
            print(
                f"Error during pattern analysis for {image_name} (attempt {attempt + 1}): {e}"
            )
            if attempt >= max_retries - 1:
                return f"Failed to analyze patterns for {image_name} after {max_retries} attempts."
            await asyncio.sleep(2)
    return f"Unknown patterns for {image_name}"


async def compare_pattern_agreement(
    reference_pattern,
    generated_pattern,
    api_key,
    model_name="gemini-2.5-flash-preview-05-20",
    max_retries=2,
):
    """Function to compare pattern descriptions and determine if they match."""
    if not api_key or not reference_pattern or not generated_pattern:
        return "No"

    client = genai.Client(api_key=api_key)

    for attempt in range(max_retries):
        try:
            prompt = f"""Compare these two pattern descriptions and determine if they represent the same or very similar patterns:

REFERENCE PATTERN:
{reference_pattern}

GENERATED PATTERN:
{generated_pattern}

Analyze if they match based on:
- Pattern type (must be the same or very similar)
- Pattern colors (should be similar or complementary)
- Orientation (should match)
- Spacing (should be similar)

Respond with ONLY "Yes" if the patterns match well, or "No" if they don't match. Be strict in your evaluation - only say "Yes" if the patterns are genuinely similar across all four attributes."""

            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model=model_name, contents=[prompt]
                ),
            )
            result = response.text.strip().lower()
            return "Yes" if "yes" in result else "No"
        except Exception as e:
            print(f"Error during pattern comparison (attempt {attempt + 1}): {e}")
            if attempt >= max_retries - 1:
                return "No"
            await asyncio.sleep(2)
    return "No"


async def segment_garment(
    pil_image,
    target_garment,
    api_key,
    model_name="gemini-2.5-flash-preview-05-20",
    max_retries=0,
    base_delay=2,
):
    """Segments the primary garment from a PIL image using the GeminiSegmentationModel."""
    if not api_key:
        return pil_image, None, "API key not provided"

    print(f"Attempting segmentation with module using {model_name}...")
    model = GeminiSegmentationModel(api_key=api_key, model_id=model_name)

    # The model's segment_image method requires a file path.
    # We'll save the PIL image to a temporary file.
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_f:
        pil_image.convert("RGB").save(temp_f.name, format="PNG")
        temp_path = temp_f.name

    try:
        loop = asyncio.get_event_loop()
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    print(f"Retry attempt {attempt}/{max_retries}...")

                # The segment_image method is synchronous, so we run it in an executor.
                segmentation_data, _ = await loop.run_in_executor(
                    None, model.segment_image, temp_path, target_garment
                )

                if segmentation_data:
                    # The module found segments. We need to combine the masks.
                    img_size_hw = (pil_image.height, pil_image.width)
                    combined_mask_np = np.zeros(img_size_hw, dtype=np.uint8)
                    labels = []
                    for mask_np, label in segmentation_data:
                        if (
                            mask_np is not None
                            and mask_np.shape == combined_mask_np.shape
                        ):
                            combined_mask_np = np.maximum(combined_mask_np, mask_np)
                        labels.append(label)

                    if np.any(combined_mask_np):
                        mask_pil = Image.fromarray(combined_mask_np, "L")
                        # Use our existing apply_mask to get a transparent background
                        segmented_pil = apply_mask(pil_image, mask_pil)
                        description = ", ".join(labels) or target_garment
                        print("✅ Segmentation successful with module.")
                        return segmented_pil, mask_pil, description

                print(
                    f"Segmentation attempt {attempt + 1} with module failed: no objects found."
                )
                if attempt >= max_retries:
                    break
                await asyncio.sleep(base_delay * (2**attempt))

            except Exception as e:
                print(f"Error during module segmentation (attempt {attempt + 1}): {e}")
                if attempt >= max_retries:
                    break
                await asyncio.sleep(base_delay * (2**attempt))
    finally:
        os.remove(temp_path)

    # If all attempts fail, use a fallback generic mask.
    print(
        "All segmentation attempts failed. Using generic segmentation as final fallback."
    )
    generic_mask = create_generic_mask(pil_image)
    generic_segmented = apply_mask(pil_image, generic_mask)
    return generic_segmented, generic_mask, "Fallback Generic Mask"


# --- Main Analysis Pipeline ---


async def perform_deterministic_analysis(
    image1_pil_orig, image2_pil_orig, image3_pil_orig, api_key
):
    """The main function to perform the full analysis pipeline."""
    st.write("### Gemini AI Segmentation Analysis")
    if not api_key:
        st.error("❌ Google AI API Key is required. Cannot proceed.")
        return None

    with st.spinner("Resizing images for consistent processing..."):
        image1_pil = resize_image_to_fixed_size(image1_pil_orig)
        image2_pil = resize_image_to_fixed_size(image2_pil_orig)
        image3_pil = resize_image_to_fixed_size(image3_pil_orig)

    with st.spinner("Step 1: Identifying reference garment..."):
        ref_desc = await identify_garment_characteristics(image1_pil, api_key)
        if "Failed" in ref_desc or "Unknown" in ref_desc:
            st.error(f"Could not identify reference garments: {ref_desc}")
            return None

    with st.expander("**Reference Garments Identified**", expanded=True):
        st.write(ref_desc)

    with st.spinner("Step 2: Analyzing patterns on all images..."):
        pattern_tasks = [
            analyze_pattern_descriptions(image1_pil, "Reference Image", api_key),
            analyze_pattern_descriptions(image2_pil, "Generated Image 1", api_key),
            analyze_pattern_descriptions(image3_pil, "Generated Image 2", api_key),
        ]
        pattern_results = await asyncio.gather(*pattern_tasks)

    with st.expander("**Pattern Analysis Results**", expanded=True):
        pattern_col1, pattern_col2, pattern_col3 = st.columns(3)

        with pattern_col1:
            st.subheader("Reference Image Patterns")
            st.write(pattern_results[0])

        with pattern_col2:
            st.subheader("Generated Image 1 Patterns")
            st.write(pattern_results[1])

        with pattern_col3:
            st.subheader("Generated Image 2 Patterns")
            st.write(pattern_results[2])

    with st.spinner("Comparing pattern agreement..."):
        agreement_tasks = [
            compare_pattern_agreement(pattern_results[0], pattern_results[1], api_key),
            compare_pattern_agreement(pattern_results[0], pattern_results[2], api_key),
        ]
        agreement_results = await asyncio.gather(*agreement_tasks)

    with st.expander("**Pattern Agreement Analysis**", expanded=True):
        agreement_col1, agreement_col2 = st.columns(2)

        with agreement_col1:
            st.subheader("Reference vs Generated 1")
            agreement_color_1 = "green" if agreement_results[0] == "Yes" else "red"
            st.markdown(
                f"**Pattern Agreement:** <span style='color:{agreement_color_1}'>{agreement_results[0]}</span>",
                unsafe_allow_html=True,
            )

        with agreement_col2:
            st.subheader("Reference vs Generated 2")
            agreement_color_2 = "green" if agreement_results[1] == "Yes" else "red"
            st.markdown(
                f"**Pattern Agreement:** <span style='color:{agreement_color_2}'>{agreement_results[1]}</span>",
                unsafe_allow_html=True,
            )

    with st.spinner(f"Step 3: Segmenting all images based on description..."):
        results = []
        for idx, img in enumerate([image1_pil, image2_pil, image3_pil], start=1):
            st.write(f"Segmenting image {idx} / 3 ...")
            try:
                res = await segment_garment(img, ref_desc, api_key)
            except Exception as e:
                res = e
            results.append(res)

    processed_results = []
    for i, res in enumerate(results):
        img_pil = [image1_pil, image2_pil, image3_pil][i]
        if isinstance(res, Exception) or res[1] is None:
            st.warning(
                f"Segmentation failed for image {i+1}. Reason: {res[2] if isinstance(res, tuple) else res}. Using original image for analysis."
            )
            processed_results.append((img_pil, None, "Fallback"))  # Use original image
        else:
            processed_results.append(res)

    segmented_img1, mask1, _ = processed_results[0]
    segmented_img2, mask2, _ = processed_results[1]
    segmented_img3, mask3, _ = processed_results[2]

    if mask1 is None:
        st.warning(
            "Could not segment reference image. Analysis will be based on the full image."
        )
        # We can still proceed using the unsegmented image.
        # The feature extraction will run on the full image.

    with st.spinner("Step 4: Extracting features and comparing..."):
        features1 = extract_features(image1_pil, mask1, "Reference")
        features2 = extract_features(image2_pil, mask2, "Generated 1")
        features3 = extract_features(image3_pil, mask3, "Generated 2")

        if not features1:
            st.error("Could not extract features from reference image. Aborting.")
            return None

        # Comparisons
        def compare_features(f1, f2, img1, img2, mask1, mask2):
            if not f2:
                return {"ciede2000": 0.0}
            c2000 = calculate_ciede2000_color_similarity(
                f1["dominant_color"], f2["dominant_color"]
            )
            return {"ciede2000": c2000}

        comp_1_2 = compare_features(
            features1, features2, image1_pil, image2_pil, mask1, mask2
        )
        comp_1_3 = compare_features(
            features1, features3, image1_pil, image3_pil, mask1, mask3
        )

    # --- Display Results ---
    st.write("---")
    st.write("### 📊 Comparison Results")

    final_images_col, scores_col = st.columns([2, 1])

    with final_images_col:
        st.image(
            [
                segmented_img1 or image1_pil,
                segmented_img2 or image2_pil,
                segmented_img3 or image3_pil,
            ],
            caption=[
                "Reference (Segmented)",
                "Generated 1 (Segmented)",
                "Generated 2 (Segmented)",
            ],
            use_column_width=True,
        )

    with scores_col:
        with st.expander("**Reference vs. Generated 1**", expanded=True):
            st.metric("CIEDE2000 Color Similarity", f"{comp_1_2['ciede2000']:.1f}%")
            agreement_color_1 = "green" if agreement_results[0] == "Yes" else "red"
            st.markdown(
                f"**Pattern Agreement:** <span style='color:{agreement_color_1}'>{agreement_results[0]}</span>",
                unsafe_allow_html=True,
            )

        st.write("---")

        with st.expander("**Reference vs. Generated 2**", expanded=True):
            st.metric("CIEDE2000 Color Similarity", f"{comp_1_3['ciede2000']:.1f}%")
            agreement_color_2 = "green" if agreement_results[1] == "Yes" else "red"
            st.markdown(
                f"**Pattern Agreement:** <span style='color:{agreement_color_2}'>{agreement_results[1]}</span>",
                unsafe_allow_html=True,
            )

    return {"comp_1_2": comp_1_2, "comp_1_3": comp_1_3}


# --- Streamlit UI ---


def main():
    st.set_page_config(layout="wide", page_title="Garment Comparison Tool")

    st.title("👕 Garment Visual Comparison Tool")
    st.write(
        "Upload a reference garment image and two generated variants to compare them using Gemini-powered segmentation and analysis."
    )

    with st.expander("About The Analysis Methods"):
        st.subheader("Dominant Color Extraction")
        st.markdown("""
        Dominant color extraction provides a high-level summary of the garment's main colors.
        - **Method**: Clustering algorithms, most commonly k-means, are applied to the pixel color values within the segmented garment region. The centroids of the resulting clusters represent the dominant colors.
        - **Tools**: Python libraries OpenCV (`cv2.kmeans`) and scikit-learn (`sklearn.cluster.KMeans`) offer robust implementations of k-means.
        - **Process**:
            1. Read the input and output images.
            2. Perform garment segmentation on both images.
            3. For each segmented garment, convert the pixel data (typically RGB values) into a list suitable for clustering.
            4. Apply k-means algorithm to find a predefined number of 'k' cluster centers (dominant colors). The choice of 'k' can be fixed or determined dynamically.
            5. The resulting 'k' centroids (e.g., in RGB or CIELAB space) from the input garment are then compared against those from the output garment. This comparison can be done by calculating perceptual color difference metrics (discussed later) between corresponding dominant colors or by assessing the similarity of the sets of dominant colors.
        - **Relevance**: This method is useful for a quick, overall assessment of color fidelity.
        """)

    with st.sidebar:
        st.header("Controls")
        google_api_key = st.text_input(
            "Google AI API Key", type="password", help="Required for Gemini analysis."
        )

        st.header("Upload Images")
        image_file_1 = st.file_uploader(
            "Reference Garment", type=["png", "jpg", "jpeg"]
        )
        image_file_2 = st.file_uploader(
            "Generated Garment 1", type=["png", "jpg", "jpeg"]
        )
        image_file_3 = st.file_uploader(
            "Generated Garment 2", type=["png", "jpg", "jpeg"]
        )

    col1, col2, col3 = st.columns(3)
    if image_file_1:
        col1.image(image_file_1, caption="Reference Image", use_column_width=True)
    if image_file_2:
        col2.image(image_file_2, caption="Generated Image 1", use_column_width=True)
    if image_file_3:
        col3.image(image_file_3, caption="Generated Image 2", use_column_width=True)

    if st.button("Run Analysis", use_container_width=True):
        if image_file_1 and image_file_2 and image_file_3:
            if google_api_key:
                image1 = Image.open(image_file_1)
                image2 = Image.open(image_file_2)
                image3 = Image.open(image_file_3)

                with st.spinner("Performing full analysis... This may take a minute."):
                    asyncio.run(
                        perform_deterministic_analysis(
                            image1, image2, image3, google_api_key
                        )
                    )
            else:
                st.error("Please provide a Google AI API Key in the sidebar.")
        else:
            st.warning("Please upload all three images.")


if __name__ == "__main__":
    main()
