# ocr_packaging.py
# Goal: Extract medicine name from packaging image using OCR

import cv2
import pytesseract
from PIL import Image
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def load_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not load: {image_path}")
        return None
    print(f"✅ Loaded — {img.shape[1]}x{img.shape[0]} pixels")
    return img


def preprocess_variants(img):
    """
    Generate multiple preprocessed versions of the image.
    We try several and OCR all of them — best result wins.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Scale up for better OCR
    scaled = cv2.resize(gray, None, fx=2, fy=2,
                        interpolation=cv2.INTER_CUBIC)

    variants = {}

    # Variant 1: Basic threshold (OTSU)
    _, otsu = cv2.threshold(scaled, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants['otsu'] = otsu

    # Variant 2: Inverted threshold
    # (useful when text is light on dark background)
    variants['otsu_inv'] = cv2.bitwise_not(otsu)

    # Variant 3: Adaptive threshold
    # (handles uneven lighting across image)
    adaptive = cv2.adaptiveThreshold(
        scaled, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10
    )
    variants['adaptive'] = adaptive

    # Variant 4: Sharpened + OTSU
    kernel = np.array([[0, -1,  0],
                       [-1, 5, -1],
                       [0, -1,  0]])
    sharpened = cv2.filter2D(scaled, -1, kernel)
    _, sharp_otsu = cv2.threshold(sharpened, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants['sharp_otsu'] = sharp_otsu

    # Variant 5: Original color image
    # (sometimes Tesseract does better on color)
    color_scaled = cv2.resize(img, None, fx=2, fy=2,
                              interpolation=cv2.INTER_CUBIC)
    variants['color'] = color_scaled

    return variants


def ocr_all_variants(variants):
    """Run OCR on all variants and collect results."""
    print("\n🔬 Running OCR on all preprocessing variants...\n")

    results = {}
    config = '--psm 6'

    for name, img_variant in variants.items():
        # Convert to PIL
        if len(img_variant.shape) == 2:
            pil_img = Image.fromarray(img_variant)
        else:
            pil_img = Image.fromarray(
                cv2.cvtColor(img_variant, cv2.COLOR_BGR2RGB)
            )

        text = pytesseract.image_to_string(pil_img, config=config)
        word_count = len(text.split())
        print(f"  [{name}] → {word_count} words")
        results[name] = text

    return results


def extract_medicine_name_from_text(text):
    """
    From raw OCR text of packaging, find the most likely medicine name.
    Strategy: longest UPPERCASE word cluster is usually the brand name.
    """
    import re

    lines = text.split('\n')
    candidates = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Skip lines that are clearly not medicine names
        skip_keywords = [
            'batch', 'mfg', 'exp', 'mrp', 'manufactured',
            'store', 'keep', 'warning', 'dosage', 'direction',
            'www', 'http', 'ltd', 'pvt', 'inc'
        ]
        if any(kw in line.lower() for kw in skip_keywords):
            continue

        # Count uppercase words in line
        words = line.split()
        upper_words = [w for w in words if w.isupper() and len(w) > 2]

        if upper_words:
            candidates.append({
                'line'        : line,
                'upper_count' : len(upper_words),
                'upper_text'  : ' '.join(upper_words)
            })

    # Sort by number of uppercase words
    candidates.sort(key=lambda x: x['upper_count'], reverse=True)

    return candidates


def run_packaging_ocr(image_path):
    """Full pipeline for packaging OCR."""
    print(f"\n📦 Running packaging OCR on: {image_path}")
    print("-" * 50)

    # 1. Load
    img = load_image(image_path)
    if img is None:
        return None

    # 2. Generate preprocessing variants
    variants = preprocess_variants(img)

    # 3. Save debug images
    for name, v in variants.items():
        cv2.imwrite(f"images/packaging/debug_{name}.jpg", v)
    print("✅ Debug variants saved → images/packaging/")

    # 4. OCR all variants
    results = ocr_all_variants(variants)

    # 5. Pick best result (most words)
    best_name = max(results, key=lambda k: len(results[k].split()))
    best_text = results[best_name]
    print(f"\n✅ Best variant: {best_name}")

    # 6. Extract medicine name candidates
    candidates = extract_medicine_name_from_text(best_text)

    # 7. Display results
    print("\n📝 FULL OCR TEXT (best variant):")
    print("-" * 50)
    print(best_text)
    print("-" * 50)

    print("\n💊 MEDICINE NAME CANDIDATES (ranked):")
    print("-" * 50)
    if candidates:
        for i, c in enumerate(candidates[:5], 1):
            print(f"  [{i}] {c['upper_text']}")
            print(f"       (from line: '{c['line']}')")
    else:
        print("  ❌ No strong candidates found")

    print("\n🏆 TOP GUESS:", candidates[0]['upper_text'] if candidates else "Unknown")

    return {
        'raw_text'   : best_text,
        'candidates' : candidates,
        'top_guess'  : candidates[0]['upper_text'] if candidates else None
    }


# ── Main ──────────────────────────────────────────────────
if __name__ == "__main__":
    result = run_packaging_ocr("images/packaging/current_package.jpg")