import io
from PIL import Image
from google import genai
from google.genai import types

def compress_image(image_path, max_dim=256, quality=50):
    """Aggressively compresses images to ensure fast network transfers and avoid SSL timeout errors."""
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()

def run_diagnostic_assistant(api_key, original_img_path, heatmap_img_path, patient_notes, cup_to_disc_score):
    """
    Calls the live Gemini API directly using active models and bypassed SSL settings.
    """
    client = genai.Client(
        api_key=api_key.strip(),
        http_options=types.HttpOptions(
            timeout=60000,  # 60 seconds
            client_args={"verify": False}
        )
    )
    
    fundus_bytes = compress_image(original_img_path)
    heatmap_bytes = compress_image(heatmap_img_path)

    fundus_part = types.Part.from_bytes(data=fundus_bytes, mime_type="image/jpeg")
    heatmap_part = types.Part.from_bytes(data=heatmap_bytes, mime_type="image/jpeg")

    prompt = f"""
    You are a supportive clinical AI assistant collaborating with an ophthalmologist. You are a helper, NOT a replacement for a doctor.
    Analyze the provided Color Fundus Photograph and the Grad-CAM Heatmap overlay focused precisely on the Optic Nerve Head.
    
    PATIENT DATA & METRICS:
    - Estimated Cup-to-Disc Ratio (CDR) Score: {cup_to_disc_score:.2f}
    - Clinical Notes: {patient_notes if patient_notes else 'None provided.'}
    
    STRICT FORMATTING & PERSONA REQUIREMENTS:
    1. You must start your response with the exact phrase: **"Doctor, consider this finding: ..."**
    2. **Confidence Level**: Explicitly state your confidence using the exact phrasing format: **"I'm [X]% sure..."** (e.g., "I'm 92% sure..." or "I'm 25% sure...").
    3. **Visual Region Analysis**: Point directly to the highlighted Optic Nerve Head area. Explain what you found by contrasting it with standard anatomy:
       - Normal Optic Nerve: small cup, nerve rim is pink and thick.
       - Glaucomatous Optic Nerve: enlarged cup, nerve rim is thin and pale.
    4. **Collaborative Next Steps**: Offer professional suggestions for secondary clinical verification (e.g., SD-OCT, SAP Visual Field testing, IOP measurement).
    """

    contents = [fundus_part, heatmap_part, prompt]
    
    # Only use active, supported model IDs
    candidate_models = ["gemini-3.8-flash", "gemini-3.5-flash"]

    last_exception = None
    for model_name in candidate_models:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents
            )
            if response and response.text:
                return response.text
        except Exception as e:
            last_exception = e
            continue
    
    raise RuntimeError(f"Live Gemini API request failed. Last error: {last_exception}")