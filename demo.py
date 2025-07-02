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
from tools.cache_helper import get_cache_content, save_cache_content
from tools.constants import BASE_EXPERIMENT_NAME
from utils import (
    apply_mask,
    calculate_ciede2000_color_similarity,
    create_generic_mask,
    parse_json_from_text,
    display_pattern_results,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_pattern_agreement_score(agreement_results):
    scores_sum = sum([garment["score"] * garment["min_mask_pixels_count"] for garment in agreement_results])
    total_pixels_count = sum([garment["min_mask_pixels_count"] for garment in agreement_results])
    average_score = scores_sum / total_pixels_count / 3 * 100
    return average_score


def pattern_agreement_display_format(agreement_results):
    output_text = ("\n" + "-" * 100 + "\n").join(
        [
            f"""
Garment: {garment["garment"]}<br/>
Agreement: {garment["agreement"]}<br/>
Reason: {garment["reason"]}<br/>
Details: {garment["details"]}
"""
            for garment in agreement_results
        ]
    )
    return output_text


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


def get_merged_mask(processed_image_obj):
    """Get the merged mask of all segmented images."""
    masks = [garment_obj['mask'] for garment_obj in processed_image_obj.values()]
    # masks are PIL mode "L" image objects
    merged_mask = np.zeros_like(masks[0])
    for mask in masks:
        merged_mask = np.logical_or(merged_mask, np.array(mask))
    return Image.fromarray(merged_mask)


# --- Gemini API and Segmentation ---


async def identify_garment_characteristics(
    pil_image, image_name, api_key, model_name="gemini-2.5-flash", max_retries=2
):
    global BASE_EXPERIMENT_NAME
    cache_parent_folder = f"{BASE_EXPERIMENT_NAME}/garment_characteristics"
    cache_response = get_cache_content(cache_parent_folder, image_name, type="json")
    if cache_response:
        print(f"Cache hit for {image_name}")
        return cache_response

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

Focus on the most visible and prominent garments. Return the output in JSON format in the form of:
{
    "garments": ["garment1", "garment2", "garment3"...]
}.

Query:"""

            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model=model_name, contents=[prompt, pil_image]
                ),
            )
            response_text = parse_json_from_text(response.text.strip())
            save_cache_content(
                cache_parent_folder, image_name, response_text, type="json"
            )
            return response_text
        except Exception as e:
            print(f"Error during garment identification (attempt {attempt + 1}): {e}")
            if attempt >= max_retries - 1:
                return f"Failed to identify garments after {max_retries} attempts."
            await asyncio.sleep(2)

    return "Unknown Garments"


async def analyze_pattern_descriptions_v2(
    pil_image,
    image_name,
    api_key,
    model_name="gemini-2.5-flash",
    max_retries=2,
    garments_list=[],
) -> list[dict]:
    """Function to analyze and describe all patterns visible in an image."""
    global BASE_EXPERIMENT_NAME
    cache_parent_folder = f"{BASE_EXPERIMENT_NAME}/pattern_analysis"
    cache_response = get_cache_content(cache_parent_folder, image_name, type="json")

    if cache_response:
        print(f"Cache hit for {image_name}")
        return cache_response

    if pil_image is None or not api_key:
        return f"Invalid input for {image_name}"

    client = genai.Client(api_key=api_key)
    results = []

    if len(garments_list) == 0:
        garments_list = [""]

    for garment in garments_list:
        prompt = (
            f"""Analyze this garment image and list the patterns visible {f"for {garment}" if garment else ""}.

    For each pattern, provide only these four attributes:
    * Pattern type: (one of the following: stripes, florals, geometric, solid, plaid, dots, checks, paisley, animal_print, tribal, abstract, botanical, damask, houndstooth, chevron, tie_dye, ombre, camouflage, argyle, toile, ikat, batik, herringbone, gingham, tartan, windowpane, pinstripe, lace, embroidered, applique, beaded, sequined, metallic, mesh, crochet, knit, quilted, textured, other)
    * Pattern colors: (list the main colors used in the pattern)
    * Orientation: (one of the following: horizontal, vertical, diagonal, random, radial, circular, spiral, concentric, symmetrical, asymmetrical, scattered, border, allover, placement, directional, mirrored, cascading, crosshatch, interlocking, overlapping, graduated, clustered, linear, grid, alternating, flowing, other)
    * Spacing: (one of the following: tight, medium, wide, irregular, dense, sparse, compact, loose, even, uneven, variable, graduated, overlapping, touching, separated, clustered, scattered, uniform, progressive, rhythmic, structured, random, other)

    If multiple distinct patterns exist, list each separately.
    Return the output in JSON format in the form of:"""
            + """
    ```json
    {
        "patterns": [
            {"pattern_type": "pattern_type", "pattern_colors": ["color1", "color2"], "orientation": "orientation", "spacing": "spacing"},
            ...
        ]
    }
    ```
    Query:
    """
        )

        for attempt in range(max_retries):
            try:
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: client.models.generate_content(
                        model=model_name, contents=[prompt, pil_image]
                    ),
                )
                results.append(
                    {
                        "garment": garment,
                        "patterns": parse_json_from_text(response.text.strip()),
                    }
                )
                break
            except Exception as e:
                print(
                    f"Error during pattern analysis for {image_name} (attempt {attempt + 1}): {e}"
                )
                if attempt >= max_retries - 1:
                    return f"Failed to analyze patterns for {image_name} after {max_retries} attempts."
                await asyncio.sleep(2)

    save_cache_content(cache_parent_folder, image_name, results, type="json")

    return results


async def analyze_pattern_descriptions(
    pil_image,
    image_name,
    api_key,
    model_name="gemini-2.5-flash",
    max_retries=2,
    garments_list=[],
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
    model_name="gemini-2.5-flash",
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


async def compare_pattern_agreement_v2(
    reference_image,
    reference_image_name,
    generated_image,
    generated_image_name,
    garment_name,
    api_key,
    model_name="gemini-2.5-flash",
    max_retries=2,
):
    """Function to compare pattern descriptions b/w reference and generated images"""
    if (
        not api_key
        or not reference_image
        or not generated_image
    ):
        return "No"

    # Make a mask of the reference image and count the number of non-transparent pixels
    reference_image_mask_pixels_count = int((np.array(reference_image)[:, :, 3] != 0).sum())

    # Make a mask of the generated image and count the number of non-transparent pixels
    generated_image_mask_pixels_count = int((np.array(generated_image)[:, :, 3] != 0).sum())

    min_mask_pixels_count = min(reference_image_mask_pixels_count, generated_image_mask_pixels_count)

    global BASE_EXPERIMENT_NAME
    cache_parent_folder = f"{BASE_EXPERIMENT_NAME}/pattern_agreement"
    cache_filename = f"{reference_image_name}_{generated_image_name}_{garment_name}"
    cache_response = get_cache_content(cache_parent_folder, cache_filename, type="json")
    if cache_response:
        print(f"Cache hit for {cache_filename}")
        return cache_response

    client = genai.Client(api_key=api_key)

    try:
        prompt = (
            f"""Compare the two images, and determine if they represent the same or very similar patterns or very different patterns:

Mark first image as REFERENCE IMAGE and second image as GENERATED IMAGE.

Focus on following garment:
- {garment_name}

Consider the image (Reference Image the ground truth) to analyze if they match based on the following attributes:
- Pattern type aspects, e.g. spacing, orientation
- Pattern colors / shininess / texture / neighbouring contrasts / etc.
- 1:1 item to color matching
- Spacing

Agreement values
- No : visually very different on all above attributes
- Partial : visually similar on some attributes
- Moderate : visually similar on most attributes
- High : visually "very" similar on all attributes (e.g. pattern type exactly following the reference image)

Respond with JSON format in the form of:"""
            + """```json
    {
        "agreement": "No/Partial/Moderate/High",
        "reason": "Reason for the agreement/disagreement",
        "details": {
            "reference_image": "Description of garment in reference image. Description of pattern style, colors, orientation, spacing, thickness in reference image. Focus only on differentiating aspects.",
            "generated_image": "Description of garment in generated image. Description of pattern style, colors, orientation, spacing, thickness in generated image. Focus only on differentiating aspects."
        }
    }
```"""
        )

        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.models.generate_content(
                model=model_name, contents=[reference_image, generated_image, prompt]
            ),
        )
        result = response.text.strip()
        output = parse_json_from_text(result)
        output["garment"] = garment_name
        score = 0
        if output["agreement"] == "No":
            score = 0
        elif output["agreement"] == "Partial":
            score = 1
        elif output["agreement"] == "Moderate":
            score = 2
        elif output["agreement"] == "High":
            score = 3
        output["score"] = score
        output["min_mask_pixels_count"] = min_mask_pixels_count
        save_cache_content(cache_parent_folder, cache_filename, output, type="json")
        return output
    except Exception as e:
        print(f"Error during pattern comparison: {e}")
        return {
            "garment": garment_name,
            "agreement": "No",
            "reason": "Failed to analyze patterns",
        }


async def segment_garment(
    pil_image,
    target_garment,
    api_key,
    model_name="gemini-2.5-flash",
    max_retries=0,
    base_delay=2,
    experiment_name="test",
    img_name="",
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
                    None,
                    model.segment_image,
                    temp_path,
                    target_garment,
                    0.1,
                    experiment_name,
                    img_name,
                )

                if segmentation_data:
                    # The module found segments. We need to combine the masks.
                    segmented_pils = {}
                    for garment_mask, garment_label in segmentation_data:
                        mask_pil = Image.fromarray(garment_mask, "L")
                        # Use our existing apply_mask to get a transparent background
                        segmented_pil = apply_mask(pil_image, mask_pil)

                        # crop the image to the size of the mask
                        segmented_pil_cropped = segmented_pil.crop(mask_pil.getbbox())
                        segmented_pils[garment_label] = {
                            "image_cropped": segmented_pil_cropped,
                            "image": segmented_pil,
                            "mask": mask_pil,
                            "label": garment_label,
                        }

                    return segmented_pils

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
    return {
        "fallback": {
            "image": generic_segmented,
            "mask": generic_mask,
            "label": "Fallback Generic Mask",
        },
    }


# --- Main Analysis Pipeline ---


async def perform_deterministic_analysis(
    image1_pil_orig,
    image2_pil_orig,
    image3_pil_orig,
    api_key,
    gemini_model="gemini-2.5-flash",
):
    """The main function to perform the full analysis pipeline."""
    images_obj = {
        "reference": {
            "original_image": image1_pil_orig,
            "name": "Reference Image",
        },
        "generated1": {
            "original_image": image2_pil_orig,
            "name": "Generated Image 1",
        },
        "generated2": {
            "original_image": image3_pil_orig,
            "name": "Generated Image 2",
        },
    }

    st.write("### Gemini AI Segmentation Analysis")
    if not api_key:
        st.error("❌ Google AI API Key is required. Cannot proceed.")
        return None

    with st.spinner("Resizing images for consistent processing..."):
        for img_obj in images_obj.values():
            img_obj["image"] = resize_image_to_fixed_size(img_obj["original_image"])

    with st.spinner("Step 1: Identifying reference garment..."):
        ref_desc = await identify_garment_characteristics(
            images_obj["reference"]["image"],
            "Reference Image",
            api_key,
            model_name=gemini_model,
        )
        if "Failed" in ref_desc or "Unknown" in ref_desc:
            st.error(f"Could not identify reference garments: {ref_desc}")
            return None

    with st.expander("**Reference Garments Identified**", expanded=True):
        garments_list = ref_desc["garments"]
        display_text = f"""Found {len(garments_list)} garments:\n\n{"\n".join([f"- {g}" for g in garments_list])}"""
        st.write(display_text)

    with st.spinner("Step 2: Analyzing patterns on all images..."):
        pattern_tasks = [
            analyze_pattern_descriptions_v2(
                images_obj["reference"]["image"],
                images_obj["reference"]["name"],
                api_key,
                model_name=gemini_model,
                garments_list=garments_list,
            ),
            analyze_pattern_descriptions_v2(
                images_obj["generated1"]["image"],
                images_obj["generated1"]["name"],
                api_key,
                model_name=gemini_model,
                garments_list=garments_list,
            ),
            analyze_pattern_descriptions_v2(
                images_obj["generated2"]["image"],
                images_obj["generated2"]["name"],
                api_key,
                model_name=gemini_model,
                garments_list=garments_list,
            ),
        ]
        pattern_results = await asyncio.gather(*pattern_tasks)

    with st.expander("**Pattern Analysis Results**", expanded=True):
        pattern_col1, pattern_col2, pattern_col3 = st.columns(3)

        with pattern_col1:
            st.subheader("Reference Image Patterns")
            st.write(display_pattern_results(pattern_results[0]))

        with pattern_col2:
            st.subheader("Generated Image 1 Patterns")
            st.write(display_pattern_results(pattern_results[1]))

        with pattern_col3:
            st.subheader("Generated Image 2 Patterns")
            st.write(display_pattern_results(pattern_results[2]))

    cache_parent_folder = f"{BASE_EXPERIMENT_NAME}/images_dump"
    processed_results_cache = get_cache_content(
        cache_parent_folder, "processed_results", type="pkl"
    )
    if processed_results_cache is None:
        with st.spinner("Step 3: Segmenting all images based on description..."):
            results = {}
            for idx, (img_obj_name, img_obj) in enumerate(images_obj.items(), start=1):
                st.write(f"Segmenting image {idx} / 3 ...")
                try:
                    res = await segment_garment(
                        img_obj["image"],
                        ref_desc,
                        api_key,
                        model_name=gemini_model,
                        experiment_name=BASE_EXPERIMENT_NAME,
                        img_name=img_obj["name"],
                    )
                except Exception as e:
                    res = e
                results[img_obj_name] = res

        processed_results = {}
        for idx, (img_obj_name, res) in enumerate(results.items()):
            img_pil = images_obj[img_obj_name]["image"]
            if isinstance(res, Exception) or "fallback" in res:
                st.warning(
                    f"Segmentation failed for image {img_obj_name}. Reason: {res["fallback"] if "fallback" in res else res}. Using original image for analysis."
                )
                processed_results[img_obj_name] = {
                    "image": img_pil,
                    "mask": None,
                    "label": "Fallback",
                }  # Use original image
            else:
                processed_results[img_obj_name] = res

        save_cache_content(
            cache_parent_folder, "processed_results", processed_results, type="pkl"
        )
    else:
        processed_results = processed_results_cache

    with st.spinner("Comparing pattern agreement..."):
        agreement_tasks_generated1 = []
        agreement_tasks_generated2 = []
        for garment_name in processed_results['reference'].keys():
            agreement_tasks_generated1 += [
                compare_pattern_agreement_v2(
                    reference_image=processed_results['reference'][garment_name]['image'],
                    reference_image_name="Reference Image",
                    generated_image=processed_results['generated1'][garment_name]['image'],
                    generated_image_name="Generated Image 1",
                    garment_name=garment_name,
                    api_key=api_key,
                    model_name=gemini_model,
                ),
            ]
            agreement_tasks_generated2 += [
                compare_pattern_agreement_v2(
                    reference_image=processed_results['reference'][garment_name]['image'],
                    reference_image_name="Reference Image",
                    generated_image=processed_results['generated2'][garment_name]['image'],
                    generated_image_name="Generated Image 2",
                    garment_name=garment_name,
                    api_key=api_key,
                    model_name=gemini_model,
                ),
            ]
        agreement_results_generated1 = await asyncio.gather(*agreement_tasks_generated1)
        agreement_results_generated2 = await asyncio.gather(*agreement_tasks_generated2)

    with st.expander("**Pattern Agreement Analysis**", expanded=True):
        agreement_col1, agreement_col2 = st.columns(2)

        with agreement_col1:
            st.subheader("Reference vs Generated 1")
            st.markdown(
                pattern_agreement_display_format(agreement_results_generated1),
                unsafe_allow_html=True,
            )

        with agreement_col2:
            st.subheader("Reference vs Generated 2")
            st.markdown(
                pattern_agreement_display_format(agreement_results_generated2),
                unsafe_allow_html=True,
            )

    merged_mask_reference = get_merged_mask(processed_results['reference'])
    merged_mask_generated1 = get_merged_mask(processed_results['generated1'])
    merged_mask_generated2 = get_merged_mask(processed_results['generated2'])
    with st.spinner("Step 4: Extracting features and comparing..."):
        features1 = extract_features(
            images_obj["reference"]["image"],
            merged_mask_reference,
            "Reference",
        )
        features2 = extract_features(
            images_obj["generated1"]["image"],
            merged_mask_generated1,
            "Generated 1",
        )
        features3 = extract_features(
            images_obj["generated2"]["image"],
            merged_mask_generated2,
            "Generated 2",
        )

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
            features1,
            features2,
            img1=images_obj["reference"]["image"],
            img2=images_obj["generated1"]["image"],
            mask1=merged_mask_reference,
            mask2=merged_mask_generated1,
        )
        comp_1_3 = compare_features(
            features1,
            features3,
            img1=images_obj["reference"]["image"],
            img2=images_obj["generated2"]["image"],
            mask1=merged_mask_reference,
            mask2=merged_mask_generated2,
        )

    # --- Display Results ---
    st.write("---")
    st.write("### 📊 Comparison Results")

    final_images_col, scores_col = st.columns([2, 1])

    with final_images_col:
        for garment_name in processed_results['reference'].keys():
            st.subheader(f"**{garment_name}**")
            st.image(
                [
                    processed_results['reference'][garment_name]['image_cropped'],
                    processed_results['generated1'][garment_name]['image_cropped'],
                    processed_results['generated2'][garment_name]['image_cropped'],
                ],
                caption=[
                    "Reference  (Segmented)",
                    "Generated 1 (Segmented)",
                    "Generated 2 (Segmented)",
                ],
                # use_column_width=True,
            )

    with scores_col:
        with st.expander("**Reference vs. Generated 1**", expanded=True):
            st.metric("CIEDE2000 Color Similarity", f"{comp_1_2['ciede2000']:.1f}%")
            st.metric("Pattern Agreement Score", f"{get_pattern_agreement_score(agreement_results_generated1):.1f}%")
            st.markdown(
                f"**Pattern Agreement Details:** \n{pattern_agreement_display_format(agreement_results_generated1)}",
                unsafe_allow_html=True,
            )

        st.write("---")

        with st.expander("**Reference vs. Generated 2**", expanded=True):
            st.metric("CIEDE2000 Color Similarity", f"{comp_1_3['ciede2000']:.1f}%")
            st.metric("Pattern Agreement Score", f"{get_pattern_agreement_score(agreement_results_generated2):.1f}%")
            st.markdown(
                f"**Pattern Agreement Details:** \n{pattern_agreement_display_format(agreement_results_generated2)}",
                unsafe_allow_html=True,
            )

    return {"comp_1_2": comp_1_2, "comp_1_3": comp_1_3}


# --- Streamlit UI ---


def main():
    global BASE_EXPERIMENT_NAME
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
        gemini_model = st.selectbox(
            "Gemini Model",
            ["gemini-2.5-flash", "gemini-2.5-pro"],
        )
        experiment_name = st.text_input(
            "Experiment Name",
            value=BASE_EXPERIMENT_NAME,
            help="Name of the experiment to save the results to.",
        )
        BASE_EXPERIMENT_NAME = experiment_name

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
                            image1,
                            image2,
                            image3,
                            google_api_key,
                            gemini_model=gemini_model,
                        )
                    )
            else:
                st.error("Please provide a Google AI API Key in the sidebar.")
        else:
            st.warning("Please upload all three images.")


if __name__ == "__main__":
    main()
