class PromptTemplates:
    """Centralized prompt templates for Gemini-based image analysis tasks."""

    # --- Garment Identification ---
    @staticmethod
    def get_garment_identification_prompt() -> str:
        """Prompt to identify garments present in an image."""
        return (
            """Identify ALL garments visible in this image. Keep descriptions short and concise. Examples:
- Blue fitted t-shirt
- Black straight-leg jeans  
- White casual sneakers

Focus on the most visible and prominent garments. Return the output in JSON format in the form of:
```json
{{
    "garments": ["garment1", "garment2", "garment3"...]
}}
```"""
        )

    @staticmethod
    def get_pattern_image_comparison_prompt(garment_name: str) -> str:
        """Prompt to compare patterns directly from reference and generated images."""
        return (
            f"""Compare the two images, and determine if they represent the same or very similar patterns or very different patterns:

First image is the REFERENCE IMAGE and second image is a GENERATED IMAGE.

Focus on following garment:
- {garment_name}

Consider the images (Reference Image the ground truth) to analyze if they match based on the following attributes:
- Pattern type aspects, e.g. spacing, orientation, design thickness
- Pattern colors saturations / texture / relative items contrast / etc.
- 1:1 item to color mapping

Agreement values
- No : visually very different on all above attributes
- Partial : visually similar on some attributes
- Moderate : visually similar on most attributes
- High : visually \"very\" similar on all attributes (e.g. pattern type exactly following the reference image)

Respond with JSON format in the form of:
```json
{{
    "agreement": "No/Partial/Moderate/High",
    "reason": "Reason for the agreement/disagreement",
    "details": {{
        "reference_image": "Description of garment in reference image, based on above attributes. Focus only on differentiating attributes.",
        "generated_image": "Description of garment in generated image, based on above attributes. Focus only on differentiating attributes."
    }}
}}
```"""
        )

    # --- Reason Gist Prompt ---
    @staticmethod
    def get_reason_gist_prompt(reason: str) -> str:
        """Prompt to extract concise bullet points from a longer reason text."""
        return (
            f"""Given the following reason, extract maximum 2 bullet points giving \"differences\" reasons.
- Each bullet point should be a not more than 4 words.
- ignore length related differences.

Return the output in JSON format in the form of: ```json
{{
    \"bullet_points\": [\"bullet_point1\", \"bullet_point2\"]
}}```

Reason: {reason}"""
        )

    # --- Segmentation Prompt ---
    @staticmethod
    def get_segmentation_prompt(object_description: str) -> str:
        """Prompt to generate segmentation masks for the given object description."""
        return (
            f"""Give the segmentation masks for {object_description}.
Output a JSON list of segmentation masks where each entry contains the 2D bounding box in the key 'box_2d',
the segmentation mask in key 'mask', and the text label in the key 'label'."""
        )

    # --- Bounding Box System Instructions ---
    @staticmethod
    def get_bounding_box_system_instructions() -> str:
        """System instructions for Gemini bounding box generation."""
        return (
            """Return bounding boxes as a JSON array with labels. Never return masks or code fencing. Limit to 25 objects.
If an object is present multiple times, name them according to their unique characteristic (colors, size, position, unique characteristics, etc..)."""
        ) 