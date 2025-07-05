from __future__ import annotations
from datetime import datetime
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
from tools.prompt_templates import PromptTemplates
from utils import (
    apply_mask,
    calculate_ciede2000_color_similarity,
    create_generic_mask,
    parse_json_from_text,
)
from typing import Any
from pydantic import BaseModel, ConfigDict, model_validator


class GarmentInfo(BaseModel):
    image: Image.Image
    mask: Image.Image
    label: str

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ImageObj(BaseModel):
    image_name: str
    image_orig: Image.Image
    image: Image.Image | None = None
    image_cropped: Image.Image | None = None
    garment_info: dict[str, GarmentInfo] = {}

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _ensure_resized(cls, self):
        if self.image is None:
            object.__setattr__(self, "image", resize_image_to_fixed_size(self.image_orig))
        return self

    @property
    def dict_key(self) -> str:
        return self.image_name.lower().replace(" ", "_")

    def get_garment_info_json(self) -> dict[str, dict[str, Any]]:
        output = {}
        for garment_label, garment_info in self.garment_info.items():
            output[garment_label] = garment_info.model_dump()
        return output


class ViewSelections(BaseModel):
    reference_image: ImageObj
    generated_image_1: ImageObj
    generated_image_2: ImageObj
    garments_list: list[str] = []
    api_key: str
    gemini_model: str = "gemini-2.5-flash"
    max_retries: int = 3

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get_images_obj(self) -> dict[str, Any]:
        return {
            "reference": {
                "original_image": self.reference_image.image_orig,
                "image": self.reference_image.image,
                "image_name": "Reference Image",
            },
            "generated1": {
                "original_image": self.generated_image_1.image_orig,
                "image": self.generated_image_1.image,
                "image_name": "Generated Image 1",
            },
            "generated2": {
                "original_image": self.generated_image_2.image_orig,
                "image": self.generated_image_2.image,
                "image_name": "Generated Image 2",
            },
        }


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_pattern_agreement_score(agreement_results: list[dict[str, float]]) -> float:
    scores_sum = sum([garment["score"] * garment["min_mask_pixels_count"] for garment in agreement_results])
    total_pixels_count = sum([garment["min_mask_pixels_count"] for garment in agreement_results])
    average_score = scores_sum / total_pixels_count / 3 * 100
    return average_score


def get_reason_gist(reason: str, api_key: str, model_name: str = "gemini-2.5-flash-lite-preview-06-17") -> list[str]:
    client = genai.Client(api_key=api_key)

    prompt = PromptTemplates.get_reason_gist_prompt(reason)

    response = client.models.generate_content(
        model=model_name, contents=[prompt]
    )
    return parse_json_from_text(response.text.strip())["bullet_points"]


def pattern_agreement_display_format(agreement_results: list[dict[str, Any]], details: bool = False) -> str:
    def get_gist(garment, details: bool = False):
        gist_text = ""
        if details:
            gist_text += f"\nDetails: {garment['details']}<br/>"

        if garment["reason_gist"]:
            gist_text += "\n" + "\n".join([f"- {point}" for point in garment["reason_gist"]])
            return f"Reason: {gist_text}"
        else:
            return gist_text

    output_text = [
            f"""
Garment: {garment["garment"]}<br/>
Agreement: {garment["agreement"]}<br/>
{get_gist(garment, details)}
"""
            for garment in agreement_results
        ]
    return "\n------------".join(output_text)


# --- Image Processing and Feature Extraction ---


def resize_image_to_fixed_size(pil_image: Image.Image, fixed_size: tuple[int, int] = (384, 512)) -> Image.Image:
    """Resizes an image to a fixed size for all processing."""
    if pil_image.size == fixed_size:
        return pil_image
    resized_image = pil_image.resize(fixed_size, Image.Resampling.LANCZOS)
    print(f"Resized image from {pil_image.size} to {fixed_size} for processing")
    return resized_image


def extract_features(
    pil_image: Image.Image | None,
    mask: Image.Image | None,
    image_name: str = "",
) -> dict[str, float | tuple[int, int, int]] | None:
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


def get_merged_mask(processed_image_obj: dict[str, dict[str, Any]]) -> Image.Image:
    """Get the merged mask of all segmented images."""
    masks = [garment_obj['mask'] for garment_obj in processed_image_obj.values()]
    # masks are PIL mode "L" image objects
    merged_mask = np.zeros_like(masks[0])
    for mask in masks:
        merged_mask = np.logical_or(merged_mask, np.array(mask))
    return Image.fromarray(merged_mask)


# --- Gemini API and Segmentation ---


async def identify_garment_characteristics(
    pil_image: Image.Image,
    image_name: str,
    api_key: str,
    model_name: str = "gemini-2.5-flash",
    max_retries: int = 2,
) -> dict[str, Any] | str:
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
            prompt = PromptTemplates.get_garment_identification_prompt()

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


async def compare_pattern_agreement(
    reference_image: Image.Image,
    reference_image_name: str,
    generated_image: Image.Image,
    generated_image_name: str,
    garment_name: str,
    api_key: str,
    model_name: str = "gemini-2.5-flash",
    max_retries: int = 2,
) -> dict[str, Any] | str:
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
        prompt = PromptTemplates.get_pattern_image_comparison_prompt(garment_name)

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
    img_obj: ImageObj,
    target_garment: Any,
    api_key: str,
    model_name: str = "gemini-2.5-flash",
    max_retries: int = 0,
    base_delay: int = 2,
    experiment_name: str = "test",
) -> dict[str, Any] | tuple[Any, ...]:
    """Segments the primary garment from a PIL image using the GeminiSegmentationModel."""
    pil_image = img_obj.image
    img_name = img_obj.dict_key
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
                    0.0,
                    experiment_name,
                    img_name,
                )

                if segmentation_data:
                    # The module found segments. We need to combine the masks.
                    for garment_mask, garment_label in segmentation_data:
                        mask_pil = Image.fromarray(garment_mask, "L")
                        # Use our existing apply_mask to get a transparent background
                        segmented_pil = apply_mask(pil_image, mask_pil)

                        # crop the image to the size of the mask
                        segmented_pil_cropped = segmented_pil.crop(mask_pil.getbbox())
                        img_obj.garment_info[garment_label] = GarmentInfo(
                            image=segmented_pil_cropped,
                            mask=mask_pil,
                            label=garment_label,
                        )

                    return img_obj.get_garment_info_json()

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


async def perform_analysis(
    view: ViewSelections,
) -> dict[str, Any] | None:
    """The main function to perform the full analysis pipeline."""
    # Extract individual selections from the unified ViewSelections object
    view_cache_parent_folder = f"{BASE_EXPERIMENT_NAME}/view_cache"
    view_cache_filename = "object_cache.pkl"
    view_cache_response = get_cache_content(view_cache_parent_folder, view_cache_filename, type="pkl")
    if view_cache_response:
        view = view_cache_response


    api_key = view.api_key
    gemini_model = view.gemini_model

    images_obj = view.get_images_obj()

    st.write("### AI Segmentation Analysis")

    if len(view.garments_list) == 0:
        with st.spinner("Step 1: Identifying reference garment..."):
            ref_desc = await identify_garment_characteristics(
                view.reference_image.image,
                "Reference Image",
                api_key,
                model_name=gemini_model,
            )
            if "Failed" in ref_desc or "Unknown" in ref_desc:
                st.error(f"Could not identify reference garments: {ref_desc}")
                return None
        view.garments_list = ref_desc["garments"]


    with st.expander("**Reference Garments Identified**", expanded=True):
        display_text = f"""Found {len(view.garments_list)} garments:\n\n{"\n".join([f"- {g}" for g in view.garments_list])}"""
        st.write(display_text)

    # with st.spinner("Step 2: Analyzing patterns on all images..."):
    #     pattern_tasks = [
    #         analyze_pattern_descriptions_v2(
    #             images_obj["reference"]["image"],
    #             images_obj["reference"]["name"],
    #             api_key,
    #             model_name=gemini_model,
    #             garments_list=garments_list,
    #         ),
    #         analyze_pattern_descriptions_v2(
    #             images_obj["generated1"]["image"],
    #             images_obj["generated1"]["name"],
    #             api_key,
    #             model_name=gemini_model,
    #             garments_list=garments_list,
    #         ),
    #         analyze_pattern_descriptions_v2(
    #             images_obj["generated2"]["image"],
    #             images_obj["generated2"]["name"],
    #             api_key,
    #             model_name=gemini_model,
    #             garments_list=garments_list,
    #         ),
    #     ]
    #     pattern_results = await asyncio.gather(*pattern_tasks)

    # with st.expander("**Pattern Analysis Results**", expanded=True):
    #     pattern_col1, pattern_col2, pattern_col3 = st.columns(3)

    #     with pattern_col1:
    #         st.subheader("Reference Image Patterns")
    #         st.write(display_pattern_results(pattern_results[0]))

    #     with pattern_col2:
    #         st.subheader("Generated Image 1 Patterns")
    #         st.write(display_pattern_results(pattern_results[1]))

    #     with pattern_col3:
    #         st.subheader("Generated Image 2 Patterns")
    #         st.write(display_pattern_results(pattern_results[2]))

    cache_parent_folder = f"{BASE_EXPERIMENT_NAME}/images_dump"
    if len(view.reference_image.garment_info) == 0:
        with st.spinner("Step 2: Segmenting all images based on description..."):
            results = {}
            for idx, img_obj in enumerate([view.reference_image, view.generated_image_1, view.generated_image_2], start=1):
                img_obj_name = img_obj.dict_key
                with st.spinner(f"Segmenting image {idx} / 3 ..."):
                    try:
                        res = await segment_garment(
                            img_obj,
                            ref_desc,
                            api_key,
                            model_name=gemini_model,
                            experiment_name=BASE_EXPERIMENT_NAME,
                        )
                    except Exception as e:
                        res = e
                    results[img_obj_name] = res
                    for garment_name, garment_data in res.items():
                        save_cache_content(cache_parent_folder, f"{img_obj_name}_{garment_name}_segmentation_output", garment_data['image'], type="png")

        processed_results = {}
        for idx, (img_obj_name, res) in enumerate(results.items()):
            processed_results[img_obj_name] = res
    else:
        processed_results = {
            "reference_image": view.reference_image.get_garment_info_json(),
            "generated_image_1": view.generated_image_1.get_garment_info_json(),
            "generated_image_2": view.generated_image_2.get_garment_info_json(),
        }

    with st.spinner("Comparing pattern agreement..."):
        agreement_tasks_generated1 = []
        agreement_tasks_generated2 = []
        for garment_name in processed_results['reference_image'].keys():
            agreement_tasks_generated1 += [
                compare_pattern_agreement(
                    reference_image=view.reference_image.garment_info[garment_name].image,
                    reference_image_name="Reference Image",
                    generated_image=view.generated_image_1.garment_info[garment_name].image,
                    generated_image_name="Generated Image 1",
                    garment_name=garment_name,
                    api_key=api_key,
                    model_name=gemini_model,
                ),
            ]
            agreement_tasks_generated2 += [
                compare_pattern_agreement(
                    reference_image=view.reference_image.garment_info[garment_name].image,
                    reference_image_name="Reference Image",
                    generated_image=view.generated_image_2.garment_info[garment_name].image,
                    generated_image_name="Generated Image 2",
                    garment_name=garment_name,
                    api_key=api_key,
                    model_name=gemini_model,
                ),
            ]
        agreement_results_generated1 = await asyncio.gather(*agreement_tasks_generated1)
        agreement_results_generated2 = await asyncio.gather(*agreement_tasks_generated2)

    
        cache_parent_folder = f"{BASE_EXPERIMENT_NAME}/images_dump"
        cache_filename = "pattern_agreement_results"
        cache_content = get_cache_content(cache_parent_folder, cache_filename, type="json")

        if cache_content is None:
            # Enrich with reason gists (only if not already cached)
            for agreement_results in [agreement_results_generated1, agreement_results_generated2]:
                for result in agreement_results:
                    if result["agreement"] != "High":
                        result["reason_gist"] = get_reason_gist(result["reason"], api_key)
                    else:
                        result["reason_gist"] = ""

            cache_payload = {
                "generated1": agreement_results_generated1,
                "generated2": agreement_results_generated2,
            }
            save_cache_content(cache_parent_folder, cache_filename, cache_payload, type="json")
        else:
            agreement_results_generated1 = cache_content.get("generated1", [])
            agreement_results_generated2 = cache_content.get("generated2", [])


    with st.expander("**Pattern Agreement Analysis**", expanded=True):
        agreement_col1, agreement_col2 = st.columns(2)

        with agreement_col1:
            st.subheader("Reference vs Generated 1")
            st.markdown(
                pattern_agreement_display_format(agreement_results_generated1, details=True),
                unsafe_allow_html=True,
            )

        with agreement_col2:
            st.subheader("Reference vs Generated 2")
            st.markdown(
                pattern_agreement_display_format(agreement_results_generated2, details=True),
                unsafe_allow_html=True,
            )

    merged_mask_reference = get_merged_mask(processed_results['reference_image'])
    merged_mask_generated1 = get_merged_mask(processed_results['generated_image_1'])
    merged_mask_generated2 = get_merged_mask(processed_results['generated_image_2'])
    with st.spinner("Step 3: Extracting features and comparing..."):
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
        def compare_features(
            f1: dict[str, Any] | None,
            f2: dict[str, Any] | None,
        ) -> dict[str, float]:
            if not f2 or not f1:
                return {"ciede2000": 0.0}
            c2000 = calculate_ciede2000_color_similarity(
                f1["dominant_color"], f2["dominant_color"]
            )
            return {"ciede2000": c2000}

        comp_1_2 = compare_features(
            features1,
            features2,
        )
        comp_1_3 = compare_features(
            features1,
            features3,
        )

    # --- Display Results ---
    st.write("---")
    st.write("### 📊 Comparison Results")

    final_images_col, scores_col = st.columns([2, 1])

    with final_images_col:
        for garment_name in processed_results['reference_image'].keys():
            st.subheader(f"**{garment_name}**")
            st.image(
                [
                    processed_results['reference_image'][garment_name]['image'],
                    processed_results['generated_image_1'][garment_name]['image'],
                    processed_results['generated_image_2'][garment_name]['image'],
                ],
                caption=[
                    "Reference   (Segmented)",
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

        with st.expander("**Reference vs. Generated 2**", expanded=True):
            st.metric("CIEDE2000 Color Similarity", f"{comp_1_3['ciede2000']:.1f}%")
            st.metric("Pattern Agreement Score", f"{get_pattern_agreement_score(agreement_results_generated2):.1f}%")
            st.markdown(
                f"**Pattern Agreement Details:** \n{pattern_agreement_display_format(agreement_results_generated2)}",
                unsafe_allow_html=True,
            )
    
    save_cache_content(view_cache_parent_folder, view_cache_filename, view, type="pkl")

    return {"comp_1_2": comp_1_2, "comp_1_3": comp_1_3}


# --- Streamlit UI ---


def main() -> None:
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
        # gemini_model = st.selectbox(
        #     "Gemini Model",
        #     ["gemini-2.5-flash", "gemini-2.5-pro"],
        # )
        gemini_model = "gemini-2.5-pro"
        experiment_name = st.text_input(
            "Experiment Name",
            value=BASE_EXPERIMENT_NAME,
            help="Name of the experiment to save the results to.",
        )
        if experiment_name:
            BASE_EXPERIMENT_NAME = experiment_name
        else:
            BASE_EXPERIMENT_NAME = "test_"+datetime.now().strftime("%Y%m%d_%H%M%S")

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
                        perform_analysis(
                            ViewSelections(
                                reference_image=ImageObj(image_orig=image1, image_name="Reference Image"),
                                generated_image_1=ImageObj(image_orig=image2, image_name="Generated Image 1"),
                                generated_image_2=ImageObj(image_orig=image3, image_name="Generated Image 2"),
                                api_key=google_api_key,
                                gemini_model=gemini_model,
                            )
                        )
                    )
            else:
                st.error("Please provide a Google AI API Key in the sidebar.")
        else:
            st.warning("Please upload all three images.")


if __name__ == "__main__":
    main()
