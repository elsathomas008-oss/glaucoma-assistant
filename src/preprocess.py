import cv2
import numpy as np

def validate_image_quality(image_path):
    """
    Checks fundus photo for blurriness and extreme illumination issues.
    """
    img = cv2.imread(image_path)
    if img is None:
        return False, "Unable to read image file."

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = np.mean(gray)

    if blur_score < 80.0:
        return False, f"Image is too blurry (Blur score: {blur_score:.1f})."
    if brightness < 30.0 or brightness > 225.0:
        return False, f"Suboptimal lighting conditions (Brightness: {brightness:.1f})."

    return True, "Quality checks passed."