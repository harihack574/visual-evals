import cv2
import numpy as np
from PIL import Image as PILImage
from matplotlib import pyplot as plt
from google import genai
import google.generativeai as genai
import base64
import json
import io
from typing import List, Tuple, Optional, Union
import os


class GeminiSegmentationModel:
    """
    A module for text-guided image segmentation using Google's Gemini API.
    
    This class provides functionality to segment objects in images based on text descriptions,
    returning both bounding boxes and segmentation masks with original colors preserved.
    """
    
    def __init__(self, api_key: Optional[str] = None, model_id: str = "gemini-2.5-flash-preview-05-20"):
        """
        Initialize the segmentation model.
        
        Args:
            api_key: Google Gemini API key. If None, will try to get from environment variable GEMINI_API_KEY
            model_id: Model ID to use for segmentation
        """
        if api_key is None:
            api_key = os.getenv('GEMINI_API_KEY')
            if api_key is None:
                raise ValueError("API key must be provided or set in GEMINI_API_KEY environment variable")
        
        self.api_key = api_key
        self.model_id = model_id
        self.client = genai.Client(api_key=api_key)
        
        # Safety settings for the model
        self.safety_settings = [
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="BLOCK_ONLY_HIGH",
            ),
        ]
        
        # System instructions for bounding box generation
        self.bounding_box_system_instructions = """
        Return bounding boxes as a JSON array with labels. Never return masks or code fencing. Limit to 25 objects.
        If an object is present multiple times, name them according to their unique characteristic (colors, size, position, unique characteristics, etc..).
        """

    def _parse_json(self, text: str) -> str:
        """Remove markdown formatting from JSON response."""
        return text.strip().removeprefix("```json").removesuffix("```")

    def _generate_mask(self, predicted_str: str, img_height: int, img_width: int) -> List[Tuple[np.ndarray, str]]:
        """
        Generate segmentation masks from API response.
        
        Args:
            predicted_str: JSON response from the API
            img_height: Height of the target image
            img_width: Width of the target image
            
        Returns:
            List of tuples containing (mask_array, label)
        """
        try:
            items = json.loads(self._parse_json(predicted_str))
            if not isinstance(items, list):
                print("Error: Parsed JSON is not a list.")
                return []
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            print(f"Problematic string snippet: {predicted_str[:200]}...")
            return []
        except Exception as e:
            print(f"An unexpected error occurred during JSON parsing: {e}")
            return []

        segmentation_data = []
        default_label = "unknown"

        for item_idx, item in enumerate(items):
            if not isinstance(item, dict) or "box_2d" not in item or "mask" not in item:
                print(f"Skipping invalid item structure at index {item_idx}: {item}")
                continue

            label = item.get("label", default_label)
            if not isinstance(label, str) or not label:
                label = default_label

            png_str = item["mask"]
            if not isinstance(png_str, str) or not png_str.startswith("data:image/png;base64,"):
                # Try to handle plain base64 strings
                if isinstance(png_str, str) and len(png_str) > 100:  # Assume it's base64 without prefix
                    print(f"Processing item {item_idx} (label: {label}) with plain base64 mask format.")
                else:
                    print(f"Skipping item {item_idx} (label: {label}) with invalid mask format.")
                    continue
            else:
                png_str = png_str.removeprefix("data:image/png;base64,")
            
            try:
                png_bytes = base64.b64decode(png_str)
                bbox_mask = cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
                if bbox_mask is None:
                    print(f"Skipping item {item_idx} (label: {label}) because mask decoding failed.")
                    continue
            except (base64.binascii.Error, ValueError, Exception) as e:
                print(f"Error decoding base64 or image data for item {item_idx} (label: {label}): {e}")
                continue

            try:
                box = item["box_2d"]
                if not isinstance(box, list) or len(box) != 4:
                    print(f"Skipping item {item_idx} (label: {label}) with invalid box_2d format: {box}")
                    continue
                y0_norm, x0_norm, y1_norm, x1_norm = map(float, box)
                abs_y0 = max(0, min(int(y0_norm / 1000.0 * img_height), img_height - 1))
                abs_x0 = max(0, min(int(x0_norm / 1000.0 * img_width), img_width - 1))
                abs_y1 = max(0, min(int(y1_norm / 1000.0 * img_height), img_height))
                abs_x1 = max(0, min(int(x1_norm / 1000.0 * img_width), img_width))
                bbox_height = abs_y1 - abs_y0
                bbox_width = abs_x1 - abs_x0
                if bbox_height <= 0 or bbox_width <= 0:
                    print(f"Skipping item {item_idx} (label: {label}) with invalid bbox dims: {box} -> ({bbox_width}x{bbox_height})")
                    continue
            except (ValueError, TypeError) as e:
                print(f"Skipping item {item_idx} (label: {label}) due to error processing box_2d: {e}")
                continue

            try:
                if bbox_mask.shape[0] > 0 and bbox_mask.shape[1] > 0:
                    resized_bbox_mask = cv2.resize(
                        bbox_mask, (bbox_width, bbox_height), interpolation=cv2.INTER_LINEAR
                    )
                else:
                    print(f"Skipping item {item_idx} (label: {label}) due to empty decoded mask before resize.")
                    continue
            except cv2.error as e:
                print(f"Error resizing mask for item {item_idx} (label: {label}): {e}")
                continue

            full_mask = np.zeros((img_height, img_width), dtype=np.uint8)
            try:
                full_mask[abs_y0:abs_y1, abs_x0:abs_x1] = resized_bbox_mask
            except ValueError as e:
                print(f"Error placing mask for item {item_idx} (label: {label}): {e}. Shape mismatch: slice=({bbox_height},{bbox_width}), resized={resized_bbox_mask.shape}. Attempting correction.")
                try:
                    resized_bbox_mask_corrected = cv2.resize(bbox_mask, (bbox_width, bbox_height), interpolation=cv2.INTER_LINEAR)
                    full_mask[abs_y0:abs_y1, abs_x0:abs_x1] = resized_bbox_mask_corrected
                    print("  -> Corrected placement.")
                except Exception as inner_e:
                    print(f"  -> Failed to correct placement: {inner_e}")
                    continue

            segmentation_data.append((full_mask, label))

        return segmentation_data

    def _create_segmentation_overlay(self, img: PILImage.Image, segmentation_data: List[Tuple[np.ndarray, str]]) -> np.ndarray:
        """
        Create segmentation overlay with original colors.
        
        Args:
            img: Original PIL Image
            segmentation_data: List of (mask, label) tuples
            
        Returns:
            Numpy array with segmented areas showing original colors
        """
        img_array = np.array(img)
        binary_mask = np.zeros(img.size[::-1], dtype=np.uint8)

        for mask, label in segmentation_data:
            if mask is not None and mask.shape == binary_mask.shape:
                binary_mask = np.maximum(binary_mask, mask)

        # Create result with original colors
        if len(img_array.shape) == 3:  # Color image
            result = np.zeros_like(img_array, dtype=np.uint8)
            # Apply mask to each color channel
            for i in range(3):
                result[:, :, i] = np.where(binary_mask > 0, img_array[:, :, i], 0)
        else:  # Grayscale image
            result = np.zeros_like(img_array, dtype=np.uint8)
            result = np.where(binary_mask > 0, img_array, 0)

        return result

    def segment_image(self, image_path: str, object_description: str, 
                     temperature: float = 0.5) -> Tuple[List[Tuple[np.ndarray, str]], np.ndarray]:
        """
        Perform segmentation on an image based on text description.
        
        Args:
            image_path: Path to the image file
            object_description: Text description of objects to segment
            temperature: Model temperature for generation
            
        Returns:
            Tuple of (segmentation_data, segmented_image_array)
            - segmentation_data: List of (mask, label) tuples
            - segmented_image_array: Image with segmented areas showing original colors
        """
        # Load image
        img = PILImage.open(image_path)
        img_height, img_width = img.size[1], img.size[0]
        
        # Create prompt
        prompt = f"Give the segmentation masks for {object_description}. Output a JSON list of segmentation masks where each entry contains the 2D bounding box in the key 'box_2d', the segmentation mask in key 'mask', and the text label in the key 'label'."
        
        # Generate content
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=[prompt, img],
            config=types.GenerateContentConfig(
                temperature=temperature,
                safety_settings=self.safety_settings,
            )
        )
        
        # Process response
        result = response.text
        segmentation_data = self._generate_mask(result, img_height=img_height, img_width=img_width)
        
        # Create segmented image
        if segmentation_data:
            segmented_image = self._create_segmentation_overlay(img, segmentation_data)
        else:
            segmented_image = np.zeros_like(np.array(img))
            
        return segmentation_data, segmented_image

    def save_results(self, original_image_path: str, segmented_image: np.ndarray, 
                    output_path: str = None, show_comparison: bool = False) -> str:
        """
        Save segmentation results and optionally display comparison.
        
        Args:
            original_image_path: Path to original image
            segmented_image: Segmented image array
            output_path: Path to save result. If None, generates automatic name
            show_comparison: Whether to display side-by-side comparison
            
        Returns:
            Path where the result was saved
        """
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(original_image_path))[0]
            output_path = f"{base_name}_segmented.png"
        
        # Save segmented result
        cv2.imwrite(output_path, cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))
        
        if show_comparison:
            # Load original image
            original_img = np.array(PILImage.open(original_image_path))
            
            # Create side-by-side comparison
            gap_width = 20
            empty_img = np.full((original_img.shape[0], gap_width, original_img.shape[2]), 255, dtype=original_img.dtype)
            combined_image = np.hstack((original_img, empty_img, segmented_image))
            
            # Display
            fig, axes = plt.subplots(1, 1, figsize=(combined_image.shape[1] / 50, combined_image.shape[0] / 50))
            axes.imshow(combined_image)
            axes.axis('off')
            axes.set_title('Original | Segmented')
            plt.show()
        
        return output_path


# Convenience function for quick usage
def segment_image(image_path: str, object_description: str, api_key: str = None, 
                 output_path: str = None, show_comparison: bool = False) -> str:
    """
    Convenience function to quickly segment an image.
    
    Args:
        image_path: Path to the image file
        object_description: Text description of objects to segment
        api_key: Gemini API key (optional if set in environment)
        output_path: Path to save result (optional)
        show_comparison: Whether to show side-by-side comparison
        
    Returns:
        Path where the segmented result was saved
    """
    model = GeminiSegmentationModel(api_key=api_key)
    segmentation_data, segmented_image = model.segment_image(image_path, object_description)
    
    if segmentation_data:
        output_path = model.save_results(image_path, segmented_image, output_path, show_comparison)
        print(f"Segmentation completed! Found {len(segmentation_data)} objects.")
        print(f"Result saved to: {output_path}")
        return output_path
    else:
        print("No segmentation results found.")
        return None 
