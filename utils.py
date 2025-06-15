from PIL import Image, ImageDraw
import numpy as np
from skimage.color import deltaE_cie76, deltaE_ciede2000, rgb2lab
from skimage.metrics import structural_similarity as ssim


def create_generic_mask(pil_image):
    """Creates a generic, centered rectangular mask as a fallback."""
    width, height = pil_image.size
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    # Create a rectangle covering the central 80% of the image
    x0, y0 = int(width * 0.1), int(height * 0.1)
    x1, y1 = int(width * 0.9), int(height * 0.9)
    draw.rectangle([x0, y0, x1, y1], fill=255)
    print("Created generic centered fallback mask.")
    return mask


def apply_mask(image, mask):
    """Applies a mask to an image."""
    if mask is None:
        return image
    if image.size != mask.size:
        mask = mask.resize(image.size, Image.Resampling.LANCZOS)
    if mask.mode != "L":
        mask = mask.convert("L")

    img_rgba = image.convert("RGBA")
    img_rgba.putalpha(mask)
    return img_rgba


def calculate_ciede1976_color_similarity(color1, color2):
    """Calculates color similarity based on CIEDE1976 delta E."""
    if color1 is None or color2 is None:
        return 0.0
    lab1 = rgb2lab(np.uint8(np.asarray([[color1]])))
    lab2 = rgb2lab(np.uint8(np.asarray([[color2]])))
    delta_e = deltaE_cie76(lab1, lab2)[0][0]
    return max(0.0, 100.0 - delta_e)


def calculate_apl_similarity_score(apl1, apl2, normalization_range=127.5):
    """Calculates similarity between two Average Pixel Luminance values."""
    if apl1 is None or apl2 is None:
        return 0.0
    similarity = max(0.0, 1.0 - (abs(apl1 - apl2) / normalization_range))
    return max(0.0, min(100.0, similarity * 100.0))


def calculate_ciede2000_color_similarity(color1_rgb, color2_rgb):
    """Calculates color similarity using the perceptually uniform CIEDE2000 metric."""
    if color1_rgb is None or color2_rgb is None:
        return 0.0
    lab1 = rgb2lab(np.uint8(np.asarray([[color1_rgb]])))
    lab2 = rgb2lab(np.uint8(np.asarray([[color2_rgb]])))
    delta_e = deltaE_ciede2000(lab1, lab2)[0][0]
    # Convert delta E to similarity (0-100, higher is more similar). Scale factor 1.5 gives a useful range.
    return max(0.0, 100.0 - (delta_e * 1.5))


def calculate_ssim_similarity(image1, image2, mask1=None, mask2=None):
    """
    Calculates Structural Similarity Index Measure (SSIM) between two images.

    Args:
        image1: First PIL image
        image2: Second PIL image
        mask1: Optional mask for first image
        mask2: Optional mask for second image

    Returns:
        Dictionary with SSIM similarity scores
    """
    if image1 is None or image2 is None:
        return {"ssim": 0.0}

    # Convert images to grayscale numpy arrays
    img1_gray = np.array(image1.convert("L"))
    img2_gray = np.array(image2.convert("L"))

    # Resize images to the same size if they differ
    if img1_gray.shape != img2_gray.shape:
        # Resize to the smaller dimensions to avoid upscaling artifacts
        target_height = min(img1_gray.shape[0], img2_gray.shape[0])
        target_width = min(img1_gray.shape[1], img2_gray.shape[1])

        img1_pil_resized = image1.resize(
            (target_width, target_height), Image.Resampling.LANCZOS
        )
        img2_pil_resized = image2.resize(
            (target_width, target_height), Image.Resampling.LANCZOS
        )

        img1_gray = np.array(img1_pil_resized.convert("L"))
        img2_gray = np.array(img2_pil_resized.convert("L"))

    # Apply masks if provided
    if mask1 is not None or mask2 is not None:
        # Create combined mask
        combined_mask = np.ones_like(img1_gray, dtype=bool)

        if mask1 is not None:
            mask1_resized = mask1.resize(
                (img1_gray.shape[1], img1_gray.shape[0]), Image.Resampling.LANCZOS
            )
            mask1_np = np.array(mask1_resized.convert("L")) > 0
            combined_mask = combined_mask & mask1_np

        if mask2 is not None:
            mask2_resized = mask2.resize(
                (img2_gray.shape[1], img2_gray.shape[0]), Image.Resampling.LANCZOS
            )
            mask2_np = np.array(mask2_resized.convert("L")) > 0
            combined_mask = combined_mask & mask2_np

        # Apply mask to both images
        img1_gray = np.where(combined_mask, img1_gray, 0)
        img2_gray = np.where(combined_mask, img2_gray, 0)

    try:
        # Calculate SSIM with appropriate parameters
        # Use a smaller window size for better performance on smaller regions
        win_size = min(7, min(img1_gray.shape[0], img1_gray.shape[1]) // 2)
        if win_size < 3:
            win_size = 3

        ssim_index = ssim(img1_gray, img2_gray, win_size=win_size, data_range=255)

        # Convert to percentage (0-100)
        ssim_percentage = max(0.0, ssim_index * 100.0)

        return {"ssim": ssim_percentage}

    except Exception as e:
        print(f"Error calculating SSIM: {e}")
        return {"ssim": 0.0}