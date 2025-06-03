import streamlit as st
from PIL import Image, ImageDraw, ImageOps # Added ImageOps for resizing
import io
import cv2 # Still useful for bitwise_and if we apply mask with OpenCV
import numpy as np # Added for NumPy
import google.generativeai as genai # Added for Gemini
import asyncio # Added for parallel execution
import json # For parsing JSON response
import base64 # For decoding base64 mask
import traceback # Ensure traceback is imported for the exception handler

def extract_dominant_colors(pil_image, k=5):
    """
    Extracts k dominant colors from a PIL Image using k-means clustering.
    Returns a list of k RGB tuples.
    """
    if pil_image is None:
        st.warning("Cannot extract dominant colors: Input image is None.")
        return []
    try:
        # Convert PIL Image to NumPy array (RGB)
        img_np = np.array(pil_image.convert('RGB'))
        
        # Reshape the image to be a list of pixels
        pixels = img_np.reshape((-1, 3))
        
        # Convert to float32 for k-means
        pixels = np.float32(pixels)
        
        # Define criteria, number of clusters (k) and apply k-means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convert centers (dominant colors) to uint8 and then to a list of RGB tuples
        centers = np.uint8(centers)
        dominant_colors = [tuple(color) for color in centers]
        return dominant_colors
    except Exception as e:
        st.warning(f"Could not extract dominant colors: {e}")
        return []

def extract_histogram_features(pil_image, num_bins=32, top_n_peaks=5):
    """
    Extracts histogram features (top N peaks for L, a, b channels) from a PIL Image in CIELAB space.
    Returns a dictionary of features.
    """
    if pil_image is None:
        st.warning("Cannot extract histogram features: Input image is None.")
        return None
    try:
        img_np_rgb = np.array(pil_image.convert('RGB'))
        img_lab = cv2.cvtColor(img_np_rgb, cv2.COLOR_RGB2LAB)
        
        features = {}
        channels = cv2.split(img_lab)
        channel_names = ['L', 'a', 'b']
        
        for i, channel in enumerate(channels):
            hist = cv2.calcHist([channel], [0], None, [num_bins], [0, 256])
            cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            
            # Get top N peaks (bin index and value)
            # Flatten, get sorted indices, and pick top N
            flat_hist = hist.flatten()
            # Ensure we don't request more peaks than bins
            current_top_n = min(top_n_peaks, len(flat_hist))
            sorted_indices = np.argsort(flat_hist)[::-1] # Sort descending by value
            
            peaks = []
            for peak_idx in range(current_top_n):
                bin_index = sorted_indices[peak_idx]
                value = flat_hist[bin_index]
                if value > 0: # Only consider bins with actual counts
                     peaks.append({"bin": int(bin_index), "value": float(value)})
            features[f"{channel_names[i]}_peaks"] = peaks
            
            # Add basic stats
            mean_val, std_dev_val = cv2.meanStdDev(channel)
            features[f"{channel_names[i]}_stats"] = {
                "mean": float(mean_val[0][0]),
                "std_dev": float(std_dev_val[0][0])
            }
            
        return features
    except Exception as e:
        st.warning(f"Could not extract histogram features: {e}")
        return None

async def segment_garment(pil_image, api_key, model_name="gemini-1.5-pro", fast_mode=False, use_downscaling=True):
    """
    Attempts to segment the primary garment from a PIL image.
    In fast_mode, skips Gemini analysis and goes directly to classical segmentation.
    """
    effective_model_name = model_name 

    if pil_image is None:
        print("Segmentation skipped: Input image is None.")
        return pil_image, None

    # Fast mode - skip Gemini entirely
    if fast_mode:
        print("Fast mode enabled - using classical segmentation directly.")
        return apply_fast_classical_segmentation(pil_image, use_downscaling)

    if not api_key:
        print(f"Google AI API key not provided for {effective_model_name}. Using classical segmentation fallback.")
        return apply_fast_classical_segmentation(pil_image, use_downscaling)

    try:
        # Configure the API key
        genai.configure(api_key=api_key)
        
        # Use GenerativeModel interface
        model = genai.GenerativeModel(model_name)
        
        print(f"Attempting garment analysis with {effective_model_name} (GenerativeModel API) for an image...")

        img_byte_arr = io.BytesIO()
        pil_image.convert("RGB").save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        prompt = """Analyze this image and identify the main piece of clothing worn by the person. 
        
        Respond with just the name of the garment (e.g., "red shirt", "blue jeans", "black dress", "white jacket").
        
        If no clear garment is visible, respond with "no garment detected"."""
        
        # Create the image part for the GenerativeModel
        image_part = {
            "mime_type": "image/png",
            "data": img_byte_arr.getvalue()
        }
        
        response = await asyncio.get_event_loop().run_in_executor(
            None, 
            lambda: model.generate_content([prompt, image_part])
        )
        
        if response.text:
            garment_description = response.text.strip().lower()
            print(f"Gemini identified garment: {garment_description}")
            
            if "no garment" in garment_description or "not" in garment_description:
                print("No garment detected by Gemini, using original image.")
                return pil_image, None
            else:
                print(f"Garment detected: {garment_description}. Applying classical segmentation.")
                return apply_fast_classical_segmentation(pil_image, use_downscaling)
        else:
            print(f"No text response from {effective_model_name}. Using classical segmentation fallback.")
            return apply_fast_classical_segmentation(pil_image, use_downscaling)

    except Exception as e:
        print(f"Error during Gemini ({effective_model_name}) analysis: {e}. Using classical segmentation fallback.")
        traceback.print_exc()
        return apply_fast_classical_segmentation(pil_image, use_downscaling)

def apply_fast_classical_segmentation(pil_image, use_downscaling=True):
    """
    Apply optimized classical computer vision segmentation.
    Uses downscaling for speed and simplified GrabCut.
    """
    try:
        original_size = pil_image.size
        
        # Downscale for faster processing
        if use_downscaling:
            max_dimension = 400  # Reduce from original size for speed
            ratio = min(max_dimension / original_size[0], max_dimension / original_size[1])
            if ratio < 1:
                new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
                pil_working = pil_image.resize(new_size, Image.Resampling.LANCZOS)
            else:
                pil_working = pil_image
        else:
            pil_working = pil_image
        
        # Convert PIL to OpenCV format
        img_np = np.array(pil_working.convert('RGB'))
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        height, width = img_cv.shape[:2]
        
        # Create a mask initialized to probable background
        mask = np.zeros((height, width), np.uint8)
        
        # Define a rectangle around the center area (where garments typically are)
        margin_x = int(width * 0.25)  # Slightly more aggressive centering
        margin_y = int(height * 0.2)
        rect = (margin_x, margin_y, width - 2*margin_x, height - 2*margin_y)
        
        # Initialize foreground and background models
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        # Apply GrabCut algorithm with fewer iterations for speed
        cv2.grabCut(img_cv, mask, rect, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_RECT)  # Reduced from 5 to 3 iterations
        
        # Create binary mask (foreground and probable foreground = 1, others = 0)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        
        # Simplified morphological operations
        kernel = np.ones((2,2), np.uint8)  # Smaller kernel for speed
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
        
        # Resize mask back to original size if we downscaled
        if use_downscaling and pil_working.size != original_size:
            mask_pil_small = Image.fromarray(mask2 * 255, mode='L')
            mask_pil = mask_pil_small.resize(original_size, Image.Resampling.NEAREST)
        else:
            mask_pil = Image.fromarray(mask2 * 255, mode='L')
        
        # Create segmented image with black background
        black_bg = Image.new("RGB", original_size, (0, 0, 0))
        segmented_image = Image.composite(pil_image.convert("RGB"), black_bg, mask_pil)
        
        processing_note = " (downscaled)" if use_downscaling else ""
        print(f"Fast classical segmentation applied successfully{processing_note}.")
        return segmented_image, mask_pil
        
    except Exception as e:
        print(f"Fast classical segmentation failed: {e}. Using original image.")
        return pil_image, None

def calculate_histogram_similarity_score(features1, features2):
    """ Parameter top_n_peaks_extracted removed as it was unused. """
    if not features1 or not features2:
        return 0
    channel_scores = []
    channel_weights = {'L': 0.4, 'a': 0.3, 'b': 0.3}
    for ch_name in ['L', 'a', 'b']:
        stats1 = features1.get(f'{ch_name}_stats')
        stats2 = features2.get(f'{ch_name}_stats')
        peaks1_list = features1.get(f'{ch_name}_peaks', [])
        peaks2_list = features2.get(f'{ch_name}_peaks', [])
        if not stats1 or not stats2:
            channel_scores.append(0)
            continue
        mean_diff = abs(stats1['mean'] - stats2['mean'])
        mean_sim = max(0.0, 1.0 - (mean_diff / 255.0))
        std_dev_diff = abs(stats1['std_dev'] - stats2['std_dev'])
        std_dev_sim = max(0.0, 1.0 - (std_dev_diff / 128.0))
        peak_bins1 = {p['bin'] for p in peaks1_list}
        peak_bins2 = {p['bin'] for p in peaks2_list}
        intersection_size = len(peak_bins1.intersection(peak_bins2))
        union_size = len(peak_bins1.union(peak_bins2))
        peak_overlap_sim = intersection_size / union_size if union_size > 0 else 0.0
        current_channel_sim = (mean_sim * 0.4) + (std_dev_sim * 0.2) + (peak_overlap_sim * 0.4)
        channel_scores.append(current_channel_sim * channel_weights[ch_name])
    overall_similarity = sum(channel_scores) * 100
    return max(0.0, min(100.0, overall_similarity))

def delta_e_cie76(lab1, lab2):
    """Calculates Delta E (CIEDE1976) between two CIELAB colors."""
    return np.sqrt(np.sum((np.array(lab1) - np.array(lab2))**2))

def calculate_dominant_color_similarity_score(palette1_rgb, palette2_rgb, significant_delta_e_threshold=50.0):
    """
    Calculates a deterministic similarity score between two dominant color palettes (lists of RGB tuples).
    Uses CIELAB Delta E for perceptual color difference.
    Returns a score between 0 and 100.
    """
    if not palette1_rgb or not palette2_rgb:
        # If one palette is empty and the other is not, it's a 0% match.
        # If both are empty, it could be argued it's a 100% match of 'nothing', but 0% is safer for feature comparison.
        return 0.0

    # Convert palettes to CIELAB
    # Individual color conversion needed for cvtColor with single pixels
    palette1_lab = [cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_RGB2LAB)[0][0] for color in palette1_rgb]
    palette2_lab = [cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_RGB2LAB)[0][0] for color in palette2_rgb]

    def get_palette_similarity(p1_lab, p2_lab):
        if not p1_lab: # If p1 is empty, similarity is 1.0 if p2 is also empty, else 0.0 (handled by initial check)
            return 1.0 if not p2_lab else 0.0
        if not p2_lab: # If p2 is empty but p1 is not, similarity is 0.0
            return 0.0
            
        total_color_similarity = 0.0
        for c1_lab in p1_lab:
            min_dist = float('inf')
            for c2_lab in p2_lab:
                dist = delta_e_cie76(c1_lab, c2_lab)
                if dist < min_dist:
                    min_dist = dist
            # Similarity: 1.0 for Delta E = 0, down to 0.0 for Delta E >= threshold
            color_sim = max(0.0, 1.0 - (min_dist / significant_delta_e_threshold))
            total_color_similarity += color_sim
        return total_color_similarity / len(p1_lab)

    # Calculate similarity from palette1 to palette2 and vice-versa
    sim_p1_to_p2 = get_palette_similarity(palette1_lab, palette2_lab)
    sim_p2_to_p1 = get_palette_similarity(palette2_lab, palette1_lab)

    # Average the two directional similarities for a final score
    final_similarity_score = (sim_p1_to_p2 + sim_p2_to_p1) / 2.0 * 100.0
    return max(0.0, min(100.0, final_similarity_score))

def calculate_apl_similarity_score(apl1, apl2, normalization_range=127.5):
    """
    Calculates a deterministic similarity score between two Average Picture Level (APL) values.
    APL is typically the mean of the L* channel (0-255 range expected here).
    normalization_range is the absolute difference at which similarity becomes 0.
    Returns a score between 0 and 100.
    """
    if apl1 is None or apl2 is None: # Should not happen if features are extracted
        return 0.0
    
    similarity = max(0.0, 1.0 - (abs(apl1 - apl2) / normalization_range))
    return max(0.0, min(100.0, similarity * 100.0))

async def perform_deterministic_analysis(image1_pil_orig, image2_pil_orig, image3_pil_orig, api_key, fast_mode=False, use_downscaling=True):
    if fast_mode:
        segmentation_method = "Fast Classical CV"
    else:
        segmentation_method = "Hybrid (Gemini + Classical CV)" if api_key else "Classical Computer Vision"
    
    st.write(f"### Garment Segmentation Stage (Using {segmentation_method})")
    
    segmentation_tasks = [
        segment_garment(image1_pil_orig, api_key=api_key, fast_mode=fast_mode, use_downscaling=use_downscaling),
        segment_garment(image2_pil_orig, api_key=api_key, fast_mode=fast_mode, use_downscaling=use_downscaling),
        segment_garment(image3_pil_orig, api_key=api_key, fast_mode=fast_mode, use_downscaling=use_downscaling)
    ]
    
    results = await asyncio.gather(*segmentation_tasks)
    
    image1_to_process, mask1 = results[0]
    image2_to_process, mask2 = results[1]
    image3_to_process, mask3 = results[2]

    # Display segmentation results
    segmentation_success_count = 0
    segmentation_details = []
    
    if mask1 is not None:
        segmentation_success_count += 1
        # Calculate segmentation coverage (percentage of image that is segmented garment)
        mask_coverage = np.mean(np.array(mask1)) / 255 * 100
        segmentation_details.append(f"Reference: {mask_coverage:.1f}% of image segmented")
    else:
        st.warning("Reference Image: Using original image (segmentation failed)")
        segmentation_details.append("Reference: Segmentation failed")
        
    if mask2 is not None:
        segmentation_success_count += 1
        mask_coverage = np.mean(np.array(mask2)) / 255 * 100
        segmentation_details.append(f"Generated 1: {mask_coverage:.1f}% of image segmented")
    else:
        st.warning("Generated Image 1: Using original image (segmentation failed)")
        segmentation_details.append("Generated 1: Segmentation failed")
        
    if mask3 is not None:
        segmentation_success_count += 1
        mask_coverage = np.mean(np.array(mask3)) / 255 * 100
        segmentation_details.append(f"Generated 2: {mask_coverage:.1f}% of image segmented")
    else:
        st.warning("Generated Image 2: Using original image (segmentation failed)")
        segmentation_details.append("Generated 2: Segmentation failed")
    
    if segmentation_success_count > 0:
        st.success(f"✅ Successfully segmented {segmentation_success_count}/3 images using {segmentation_method}")
        # Show segmentation details in an expander
        with st.expander("📊 Segmentation Details"):
            for detail in segmentation_details:
                st.write(f"• {detail}")
    else:
        st.warning("⚠️ All segmentation attempts failed. Analysis will proceed with original images.")

    st.write("### Feature Extraction Stage")
    st.write("Extracting dominant colors from processed images...")
    ref_dom_colors = extract_dominant_colors(image1_to_process, k=5)
    gen1_dom_colors = extract_dominant_colors(image2_to_process, k=5)
    gen2_dom_colors = extract_dominant_colors(image3_to_process, k=5)

    st.write("Extracting CIELAB histogram features (including L* mean for APL) from processed images...")
    ref_hist_features = extract_histogram_features(image1_to_process)
    gen1_hist_features = extract_histogram_features(image2_to_process)
    gen2_hist_features = extract_histogram_features(image3_to_process)

    results_gen1 = {}
    results_gen2 = {}
    ref_apl = None 
    if ref_hist_features:
        ref_apl = ref_hist_features.get('L_stats', {}).get('mean')
        if ref_apl is None:
            st.error("Reference APL (L* mean) could not be extracted from (segmented) histogram features.")
    else:
        st.error("Reference histogram features (incl. APL) could not be extracted from (segmented) image. Analysis may be incomplete.")

    # --- Process Generated Image 1 ---
    if ref_dom_colors and gen1_dom_colors:
        dom_score1 = calculate_dominant_color_similarity_score(ref_dom_colors, gen1_dom_colors)
        results_gen1["dominant_color_evaluation"] = {"percentage_match": dom_score1, "is_match_human_perception": dom_score1 >= 85.0}
    else:
        st.warning("Could not calculate dominant color score for Gen 1 (ref vs gen1) due to missing features.")
        results_gen1["dominant_color_evaluation"] = {"percentage_match": 0, "is_match_human_perception": False}

    if ref_hist_features and gen1_hist_features:
        hist_score1 = calculate_histogram_similarity_score(ref_hist_features, gen1_hist_features)
        results_gen1["histogram_feature_evaluation"] = {"percentage_match": hist_score1, "is_match_human_perception": hist_score1 >= 85.0}
    else:
        st.warning("Could not calculate histogram feature score for Gen 1 (ref vs gen1) due to missing features.")
        results_gen1["histogram_feature_evaluation"] = {"percentage_match": 0, "is_match_human_perception": False}
    
    gen1_apl = gen1_hist_features.get('L_stats', {}).get('mean') if gen1_hist_features else None
    if ref_apl is not None and gen1_apl is not None:
        apl_score1 = calculate_apl_similarity_score(ref_apl, gen1_apl)
        results_gen1["luminance_apl_evaluation"] = {"percentage_match": apl_score1, "is_match_human_perception": apl_score1 >= 85.0}
    else:
        st.warning("Could not calculate APL score for Gen 1 (ref vs gen1) due to missing APL values.")
        results_gen1["luminance_apl_evaluation"] = {"percentage_match": 0, "is_match_human_perception": False}

    # --- Process Generated Image 2 ---
    if ref_dom_colors and gen2_dom_colors:
        dom_score2 = calculate_dominant_color_similarity_score(ref_dom_colors, gen2_dom_colors)
        results_gen2["dominant_color_evaluation"] = {"percentage_match": dom_score2, "is_match_human_perception": dom_score2 >= 85.0}
    else:
        st.warning("Could not calculate dominant color score for Gen 2 (ref vs gen2) due to missing features.")
        results_gen2["dominant_color_evaluation"] = {"percentage_match": 0, "is_match_human_perception": False}

    if ref_hist_features and gen2_hist_features:
        hist_score2 = calculate_histogram_similarity_score(ref_hist_features, gen2_hist_features)
        results_gen2["histogram_feature_evaluation"] = {"percentage_match": hist_score2, "is_match_human_perception": hist_score2 >= 85.0}
    else:
        st.warning("Could not calculate histogram feature score for Gen 2 (ref vs gen2) due to missing features.")
        results_gen2["histogram_feature_evaluation"] = {"percentage_match": 0, "is_match_human_perception": False}

    gen2_apl = gen2_hist_features.get('L_stats', {}).get('mean') if gen2_hist_features else None
    if ref_apl is not None and gen2_apl is not None:
        apl_score2 = calculate_apl_similarity_score(ref_apl, gen2_apl)
        results_gen2["luminance_apl_evaluation"] = {"percentage_match": apl_score2, "is_match_human_perception": apl_score2 >= 85.0}
    else:
        st.warning("Could not calculate APL score for Gen 2 (ref vs gen2) due to missing APL values.")
        results_gen2["luminance_apl_evaluation"] = {"percentage_match": 0, "is_match_human_perception": False}
        
    return {
        "generation1": results_gen1, 
        "generation2": results_gen2,
        "processed_images": [image1_to_process, image2_to_process, image3_to_process],
        "masks": [mask1, mask2, mask3]
    }

# Helper functions should be defined here, in the global scope
def display_summary_metric(label, eval_data, metric_key="percentage_match"):
    if eval_data and isinstance(eval_data.get(metric_key), (int, float)):
        st.write(f"**{label}:** {eval_data[metric_key]:.2f}%")
    else:
        st.write(f"**{label}:** N/A (analysis incomplete or data missing)")

def get_average_score(analysis_data):
    if not analysis_data: return -1
    scores = []
    dom_eval = analysis_data.get("dominant_color_evaluation")
    if dom_eval and isinstance(dom_eval.get("percentage_match"), (int, float)):
        scores.append(dom_eval.get("percentage_match"))
    hist_eval = analysis_data.get("histogram_feature_evaluation")
    if hist_eval and isinstance(hist_eval.get("percentage_match"), (int, float)):
        scores.append(hist_eval.get("percentage_match"))
    apl_eval = analysis_data.get("luminance_apl_evaluation")
    if apl_eval and isinstance(apl_eval.get("percentage_match"), (int, float)):
        scores.append(apl_eval.get("percentage_match"))
    return sum(scores) / len(scores) if scores else -1

def main():
    st.set_page_config(layout="wide") # Use wide layout
    st.title("Advanced Garment Comparison Tool")

    st.sidebar.header("API Configuration")
    api_key = st.sidebar.text_input("Enter Google AI API Key", type="password")
    
    st.sidebar.header("Segmentation Settings")
    fast_mode = st.sidebar.checkbox("Fast Mode", value=True, help="Skip Gemini analysis and use only classical segmentation for speed")
    use_downscaling = st.sidebar.checkbox("Use Downscaling", value=True, help="Process smaller images for faster segmentation")
    
    st.sidebar.info(
        "This tool uses a hybrid approach for garment segmentation. "
        "Fast Mode skips Gemini analysis and uses only classical segmentation for speed. "
        "Downscaling processes smaller images for faster results."
    )

    st.write("""
    Upload three images to compare: one reference outfit and two generated versions.
    **Segmentation Method**: The tool uses a hybrid approach:
    1. **With API Key**: Gemini identifies the garment, then applies classical CV segmentation
    2. **Without API Key**: Uses classical computer vision segmentation (GrabCut algorithm)
    
    The deterministic analysis then proceeds based on the segmented images for:
    1.  **Dominant Color Palette Match**
    2.  **CIELAB Histogram Feature Match**
    3.  **Luminance (APL) Match**
    """)
    
    st.warning("⚠️ **Deterministic Analysis with Segmentation**: This tool provides quantitative scores on segmented garments. Segmentation may not be perfect. Interpret results based on scores and visual inspection.")
    
    uploaded_file1 = st.file_uploader("Choose Reference Garment", type=['png', 'jpg', 'jpeg'], key="file1")
    uploaded_file2 = st.file_uploader("Choose Generated Garment 1", type=['png', 'jpg', 'jpeg'], key="file2")
    uploaded_file3 = st.file_uploader("Choose Generated Garment 2", type=['png', 'jpg', 'jpeg'], key="file3")
    
    if uploaded_file1 and uploaded_file2 and uploaded_file3:
        st.header("Original Uploaded Images")
        col1_orig, col2_orig, col3_orig = st.columns(3)
        
        image1_pil_orig = Image.open(uploaded_file1)
        with col1_orig:
            st.write("Reference Garment")
            st.image(image1_pil_orig, use_column_width=True)
        
        image2_pil_orig = Image.open(uploaded_file2)
        with col2_orig:
            st.write("Generated Garment 1")
            st.image(image2_pil_orig, use_column_width=True)
            
        image3_pil_orig = Image.open(uploaded_file3)
        with col3_orig:
            st.write("Generated Garment 2")
            st.image(image3_pil_orig, use_column_width=True)
        
        if st.button("Compare Garments"): 
            if fast_mode:
                segmentation_method = "Fast Classical CV"
            else:
                segmentation_method = "Hybrid (Gemini + Classical CV)" if api_key else "Classical Computer Vision"
            
            processing_note = ""
            if fast_mode:
                processing_note = " (Fast Mode - Direct Classical Segmentation)"
            elif use_downscaling:
                processing_note = " (with Downscaling for Speed)"
            
            with st.spinner(f"Performing analysis using {segmentation_method} segmentation{processing_note}..."):
                # Run the async function using asyncio.run()
                analysis_results = asyncio.run(perform_deterministic_analysis(
                    image1_pil_orig, image2_pil_orig, image3_pil_orig, 
                    api_key=api_key, fast_mode=fast_mode, use_downscaling=use_downscaling
                ))
                
                processed_images = analysis_results.get("processed_images")
                if processed_images and len(processed_images) == 3:
                    st.header(f"🔍 Segmentation Results ({segmentation_method})")
                    
                    # Display original vs segmented comparison
                    st.subheader("Original vs Segmented Comparison")
                    
                    for i, (original, segmented) in enumerate(zip([image1_pil_orig, image2_pil_orig, image3_pil_orig], processed_images)):
                        labels = ["Reference Garment", "Generated Garment 1", "Generated Garment 2"]
                        
                        st.write(f"**{labels[i]}**")
                        col_orig, col_seg = st.columns(2)
                        
                        with col_orig:
                            st.write("Original")
                            st.image(original, use_column_width=True)
                        
                        with col_seg:
                            st.write("Segmented")
                            st.image(segmented, use_column_width=True)
                        
                        st.write("---")  # Separator line
                    
                    # Also show all segmented images in a row for easy comparison
                    st.subheader("Segmented Images for Analysis")
                    col1_proc, col2_proc, col3_proc = st.columns(3)
                    with col1_proc:
                        st.write("Reference (Segmented)")
                        st.image(processed_images[0], use_column_width=True)
                    with col2_proc:
                        st.write("Generated 1 (Segmented)")
                        st.image(processed_images[1], use_column_width=True)
                    with col3_proc:
                        st.write("Generated 2 (Segmented)")
                        st.image(processed_images[2], use_column_width=True)
                else:
                    st.warning("Could not retrieve processed images for display.")

                masks = analysis_results.get("masks")
                if masks and any(mask is not None for mask in masks): 
                    st.header(f"🎭 Segmentation Masks ({segmentation_method})")
                    st.write("*White areas show the detected garment region*")
                    col1_mask, col2_mask, col3_mask = st.columns(3)
                    titles = ["Reference Mask", "Generated 1 Mask", "Generated 2 Mask"]
                    for i, mask_img in enumerate(masks):
                        with [col1_mask, col2_mask, col3_mask][i]:
                            st.write(titles[i])
                            if mask_img:
                                st.image(mask_img, use_column_width=True, channels="L")
                            else:
                                st.caption("Segmentation failed - using original image for analysis.")
                else: 
                    st.info("Segmentation masks are not available for display.")
                
                st.write("## Comparison Results - Generated Image 1")
                gen1_analysis_data = analysis_results.get("generation1")
                if gen1_analysis_data:
                    dom_eval_g1 = gen1_analysis_data.get("dominant_color_evaluation")
                    if dom_eval_g1:
                        st.write("**Dominant Color Palette Match**")
                        percentage_dom_g1 = dom_eval_g1.get("percentage_match", 0.0)
                        match_dom_g1 = dom_eval_g1.get("is_match_human_perception", False)
                        score_dom_g1_text = f"{percentage_dom_g1:.2f}" if isinstance(percentage_dom_g1, (float, int)) else "N/A"
                        if match_dom_g1:
                            st.success(f"✓ Similar ({score_dom_g1_text}% score)")
                        else:
                            st.error(f"✗ Different ({score_dom_g1_text}% score)")
                        if isinstance(percentage_dom_g1, (float, int)): st.progress(percentage_dom_g1 / 100)
                    else:
                        st.warning("Dominant color evaluation data missing for Generated Image 1.")
                    hist_eval_g1 = gen1_analysis_data.get("histogram_feature_evaluation")
                    if hist_eval_g1:
                        st.write("**Histogram Feature Match**")
                        percentage_hist_g1 = hist_eval_g1.get("percentage_match", 0.0)
                        match_hist_g1 = hist_eval_g1.get("is_match_human_perception", False)
                        score_hist_g1_text = f"{percentage_hist_g1:.2f}" if isinstance(percentage_hist_g1, (float, int)) else "N/A"
                        if match_hist_g1:
                            st.success(f"✓ Similar ({score_hist_g1_text}% score)")
                        else:
                            st.error(f"✗ Different ({score_hist_g1_text}% score)")
                        if isinstance(percentage_hist_g1, (float, int)): st.progress(percentage_hist_g1 / 100)
                    else:
                        st.warning("Histogram feature evaluation data missing for Generated Image 1.")
                    apl_eval_g1 = gen1_analysis_data.get("luminance_apl_evaluation")
                    if apl_eval_g1:
                        st.write("**Luminance (APL) Match**")
                        percentage_apl_g1 = apl_eval_g1.get("percentage_match", 0.0) 
                        match_apl_g1 = apl_eval_g1.get("is_match_human_perception", False)
                        score_apl_g1_text = f"{percentage_apl_g1:.2f}" if isinstance(percentage_apl_g1, (float, int)) else "N/A"
                        if match_apl_g1:
                            st.success(f"✓ Similar Brightness ({score_apl_g1_text}% score)")
                        else:
                            st.error(f"✗ Different Brightness ({score_apl_g1_text}% score)")
                        if isinstance(percentage_apl_g1, (float, int)): 
                            st.progress(percentage_apl_g1 / 100)
                    else:
                        st.warning("Luminance (APL) evaluation data missing for Generated Image 1.")
                else:
                    st.error("Analysis for Generated Image 1 failed or was incomplete.")

                st.write("## Comparison Results - Generated Image 2")
                gen2_analysis_data = analysis_results.get("generation2")
                if gen2_analysis_data:
                    dom_eval_g2 = gen2_analysis_data.get("dominant_color_evaluation")
                    if dom_eval_g2:
                        st.write("**Dominant Color Palette Match**")
                        percentage_dom_g2 = dom_eval_g2.get("percentage_match", 0.0)
                        match_dom_g2 = dom_eval_g2.get("is_match_human_perception", False)
                        score_dom_g2_text = f"{percentage_dom_g2:.2f}" if isinstance(percentage_dom_g2, (float, int)) else "N/A"
                        if match_dom_g2:
                            st.success(f"✓ Similar ({score_dom_g2_text}% score)")
                        else:
                            st.error(f"✗ Different ({score_dom_g2_text}% score)")
                        if isinstance(percentage_dom_g2, (float, int)): st.progress(percentage_dom_g2 / 100)
                    else:
                        st.warning("Dominant color evaluation data missing for Generated Image 2.")
                    hist_eval_g2 = gen2_analysis_data.get("histogram_feature_evaluation")
                    if hist_eval_g2:
                        st.write("**Histogram Feature Match**")
                        percentage_hist_g2 = hist_eval_g2.get("percentage_match", 0.0)
                        match_hist_g2 = hist_eval_g2.get("is_match_human_perception", False)
                        score_hist_g2_text = f"{percentage_hist_g2:.2f}" if isinstance(percentage_hist_g2, (float, int)) else "N/A"
                        if match_hist_g2:
                            st.success(f"✓ Similar ({score_hist_g2_text}% score)")
                        else:
                            st.error(f"✗ Different ({score_hist_g2_text}% score)")
                        if isinstance(percentage_hist_g2, (float, int)): st.progress(percentage_hist_g2 / 100)
                    else:
                        st.warning("Histogram feature evaluation data missing for Generated Image 2.")
                    apl_eval_g2 = gen2_analysis_data.get("luminance_apl_evaluation")
                    if apl_eval_g2:
                        st.write("**Luminance (APL) Match**")
                        percentage_apl_g2 = apl_eval_g2.get("percentage_match", 0.0)
                        match_apl_g2 = apl_eval_g2.get("is_match_human_perception", False)
                        score_apl_g2_text = f"{percentage_apl_g2:.2f}" if isinstance(percentage_apl_g2, (float, int)) else "N/A"
                        if match_apl_g2:
                            st.success(f"✓ Similar Brightness ({score_apl_g2_text}% score)")
                        else:
                            st.error(f"✗ Different Brightness ({score_apl_g2_text}% score)")
                        if isinstance(percentage_apl_g2, (float, int)): 
                            st.progress(percentage_apl_g2 / 100)
                    else:
                        st.warning("Luminance (APL) evaluation data missing for Generated Image 2.")
                else:
                    st.error("Analysis for Generated Image 2 failed or was incomplete.")
                
                st.write("## Summary of Deterministic Metrics")
                if gen1_analysis_data and gen2_analysis_data:
                    st.subheader("Generated Image 1 vs. Reference")
                    display_summary_metric("Dominant Color Palette Match", gen1_analysis_data.get("dominant_color_evaluation"))
                    display_summary_metric("Histogram Feature Match", gen1_analysis_data.get("histogram_feature_evaluation"))
                    display_summary_metric("Luminance (APL) Match", gen1_analysis_data.get("luminance_apl_evaluation"))
                    st.write("---") 
                    st.subheader("Generated Image 2 vs. Reference")
                    display_summary_metric("Dominant Color Palette Match", gen2_analysis_data.get("dominant_color_evaluation"))
                    display_summary_metric("Histogram Feature Match", gen2_analysis_data.get("histogram_feature_evaluation"))
                    display_summary_metric("Luminance (APL) Match", gen2_analysis_data.get("luminance_apl_evaluation"))
                    st.write("---")
                    st.write("### Comparison Insight (Based on Average of All Scores)")
                    avg_score_g1 = get_average_score(gen1_analysis_data)
                    avg_score_g2 = get_average_score(gen2_analysis_data)
                    if avg_score_g1 == -1 and avg_score_g2 == -1:
                        st.info("Average scores not available to provide a comparison insight.")
                    elif avg_score_g1 > avg_score_g2:
                        st.info(f"Generated Image 1 has a higher average score ({avg_score_g1:.2f}%) than Generated Image 2 ({avg_score_g2:.2f}%).")
                    elif avg_score_g2 > avg_score_g1:
                        st.info(f"Generated Image 2 has a higher average score ({avg_score_g2:.2f}%) than Generated Image 1 ({avg_score_g1:.2f}%).")
                    elif avg_score_g1 != -1: 
                         st.info(f"Both generated images have an equal average score ({avg_score_g1:.2f}%).")
                    else: 
                        st.info("Could not provide a comparison insight based on average scores.")
                else:
                    st.info("Full summary metrics cannot be displayed as analysis data for one or both generated images is missing or incomplete.")

if __name__ == "__main__":
    main()
