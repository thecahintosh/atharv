# ocr_prescription.py — Updated with ROI cropping + sharpening + multi-PSM testing
import cv2
import pytesseract
from PIL import Image
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def load_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not load image from: {image_path}")
        return None
    print(f"✅ Image loaded — size: {img.shape[1]}x{img.shape[0]} pixels")
    return img


def select_roi(img):
    """
    Let user manually draw a box around the medicine table.
    Press ENTER or SPACE to confirm selection.
    Press C to cancel and retry.
    """
    print("\n📌 A window will open.")
    print("   → Draw a box around the MEDICINE TABLE only")
    print("   → Press ENTER or SPACE to confirm")
    print("   → Press C to cancel and redraw\n")

    clone = img.copy()
    roi = cv2.selectROI(
        "Select Medicine Table — Press ENTER to confirm",
        clone,
        fromCenter=False,
        showCrosshair=True
    )
    cv2.destroyAllWindows()

    x, y, w, h = roi
    if w == 0 or h == 0:
        print("❌ No region selected. Please try again.")
        return None

    cropped = img[y:y+h, x:x+w]
    print(f"✅ Region selected: x={x}, y={y}, w={w}, h={h}")
    return cropped


def preprocess_image(img):
    """Clean and sharpen image for better OCR."""

    # Step 1: Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 2: Resize — larger scale for better OCR on small text
    scale_factor = 3
    resized = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor,
                         interpolation=cv2.INTER_CUBIC)

    # Step 3: Denoise
    denoised = cv2.fastNlMeansDenoising(resized, h=20)

    # Step 4: Sharpen
    kernel = np.array([[0, -1,  0],
                       [-1, 5, -1],
                       [0, -1,  0]])
    sharpened = cv2.filter2D(denoised, -1, kernel)

    # Step 5: Threshold
    _, thresh = cv2.threshold(sharpened, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def extract_text(preprocessed_img):
    """Run Tesseract OCR with multiple configs and pick best result."""
    pil_img = Image.fromarray(preprocessed_img)

    configs = {
        'psm6 (uniform block)' : '--psm 6',
        'psm4 (single column)' : '--psm 4',
        'psm11 (sparse text)'  : '--psm 11',
        'psm3 (auto)'          : '--psm 3',
    }

    print("\n🔬 Trying multiple OCR configs...\n")
    results = {}
    for name, config in configs.items():
        text = pytesseract.image_to_string(pil_img, config=config)
        word_count = len(text.split())
        print(f"  [{name}] → {word_count} words extracted")
        results[name] = text

    # Pick the config that extracted the most words
    best = max(results, key=lambda k: len(results[k].split()))
    print(f"\n✅ Best config: {best}")
    return results[best]


def run_ocr_pipeline(image_path):
    print(f"\n📄 Running OCR on: {image_path}")
    print("-" * 50)

    # 1. Load
    img = load_image(image_path)
    if img is None:
        return None

    # 2. Select medicine table region
    cropped = select_roi(img)
    if cropped is None:
        return None

    # 3. Save crop for inspection
    cv2.imwrite("images/prescriptions/debug_cropped.jpg", cropped)
    print("✅ Cropped region saved → debug_cropped.jpg")

    # 4. Preprocess
    preprocessed = preprocess_image(cropped)
    cv2.imwrite("images/prescriptions/debug_preprocessed.jpg", preprocessed)
    print("✅ Preprocessed image saved → debug_preprocessed.jpg")

    # 5. OCR with best config
    raw_text = extract_text(preprocessed)
    print("\n📝 RAW EXTRACTED TEXT (Best Config):")
    print("-" * 50)
    print(raw_text)
    print("-" * 50)

    return raw_text


if __name__ == "__main__":
    image_path = "images/prescriptions/prescription1.png"
    raw_text = run_ocr_pipeline(image_path)