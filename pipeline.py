# pipeline.py — Final robust version
# Fixes: lowercase o/d in dosage, confidence reporting, step numbering

import cv2
import pytesseract
from PIL import Image
import numpy as np
import re
from fuzzywuzzy import fuzz

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ──────────────────────────────────────────────
# MODULE 1: Prescription OCR
# ──────────────────────────────────────────────

def preprocess_for_prescription(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scaled = cv2.resize(gray, None, fx=3, fy=3,
                        interpolation=cv2.INTER_CUBIC)
    denoised = cv2.fastNlMeansDenoising(scaled, h=20)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    _, thresh = cv2.threshold(sharpened, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def select_roi(img):
    print("\n📌 Window opening — draw box around MEDICINE TABLE only")
    print("   Press ENTER to confirm, C to redraw\n")
    roi = cv2.selectROI(
        "Select Medicine Table — Press ENTER to confirm",
        img.copy(), fromCenter=False, showCrosshair=True
    )
    cv2.destroyAllWindows()
    x, y, w, h = roi
    if w == 0 or h == 0:
        return None
    return img[y:y+h, x:x+w]


def ocr_prescription(image_path):
    print("\n[STEP 1] 📄 Processing prescription image...")
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Cannot load: {image_path}")
        return None

    cropped = select_roi(img)
    if cropped is None:
        print("❌ No region selected")
        return None

    preprocessed = preprocess_for_prescription(cropped)
    pil_img = Image.fromarray(preprocessed)
    text = pytesseract.image_to_string(pil_img, config='--psm 6')
    print("✅ Prescription OCR complete")
    return text


# ──────────────────────────────────────────────
# MODULE 2: Prescription Parser
# ──────────────────────────────────────────────

def clean_text(raw_text):
    def fix_dosage_pattern(text):
        """
        FIX BUG 1: Handle ALL common OCR misreads in dosage pattern.
        Covers: O→0, o→0, D→0, d→0, I→1, l→1
        Pattern: X-X-X where X is a digit or common misread character
        """
        def replacer(m):
            def fix_char(c):
                # Characters commonly misread as 0
                if c in 'OoDd':
                    return '0'
                # Characters commonly misread as 1
                if c in 'IiLl':
                    return '1'
                return c
            g1 = fix_char(m.group(1))
            g2 = fix_char(m.group(2))
            g3 = fix_char(m.group(3))
            return f"{g1}-{g2}-{g3}"

        return re.sub(
            r'\b([OoDd0-9IiLl])\s*-\s*([OoDd0-9IiLl])\s*-\s*([OoDd0-9IiLl])\b',
            replacer,
            text
        )

    raw_text = fix_dosage_pattern(raw_text)

    replacements = {
        'Daity': 'Daily', 'Datty': 'Daily', 'Daiiy': 'Daily',
        'Woeks': 'Weeks', 'Weoks': 'Weeks',
        'FOLLO\\Y': 'FOLLOW',
        'Tatar': 'After', 'Atter': 'After',
        'sutete': 'sulfate',
        '"': '', "'": '', '=': '',
    }
    for wrong, correct in replacements.items():
        raw_text = raw_text.replace(wrong, correct)

    return raw_text


def parse_medicines(raw_text):
    cleaned = clean_text(raw_text)
    medicines = []
    current_medicine = None

    for line in cleaned.split('\n'):
        line = line.strip()
        if not line:
            continue

        medicine_start = re.match(r'^[\d]+[\)\.\}]\s*(.+)', line)
        if medicine_start:
            if current_medicine:
                medicines.append(current_medicine)

            content = medicine_start.group(1).strip()
            dosage_match = re.search(r'(\d+\s*-\s*\d+\s*-\s*\d+)', content)

            if dosage_match:
                dosage_raw  = dosage_match.group(1)
                dosage      = re.sub(r'\s*-\s*', '-', dosage_raw)
                dosage_pos  = content.index(dosage_raw)
                name        = content[:dosage_pos].strip().rstrip('.')
                timing      = content[dosage_pos + len(dosage_raw):].strip().lstrip('-').strip()
            else:
                parts  = re.split(r'\s{2,}', content)
                name   = parts[0].strip() if parts else content
                timing = ' '.join(parts[1:]).strip() if len(parts) >= 2 else "Not found"
                dosage = "Not found"

            current_medicine = {
                'name'   : name,
                'dosage' : dosage,
                'timing' : timing,
                'notes'  : []
            }

        elif current_medicine and re.match(
                r'^(Notes?|Composition|Compos)', line, re.IGNORECASE):
            current_medicine['notes'].append(line.strip())

    if current_medicine:
        medicines.append(current_medicine)

    return medicines


# ──────────────────────────────────────────────
# MODULE 3: Packaging OCR
# ──────────────────────────────────────────────

def ocr_packaging(image_path):
    print("\n[STEP 3] 📦 Processing medicine packaging image...")
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Cannot load: {image_path}")
        return None

    color_scaled = cv2.resize(img, None, fx=2, fy=2,
                              interpolation=cv2.INTER_CUBIC)
    pil_img = Image.fromarray(cv2.cvtColor(color_scaled, cv2.COLOR_BGR2RGB))
    text = pytesseract.image_to_string(pil_img, config='--psm 6')

    candidates = []
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        skip = ['batch', 'mfg', 'exp', 'mrp', 'store',
                'keep', 'warning', 'www', 'ltd', 'pvt']
        if any(kw in line.lower() for kw in skip):
            continue
        upper_words = [w for w in line.split()
                       if w.isupper() and len(w) > 2]
        if upper_words:
            candidates.append(' '.join(upper_words))

    if not candidates:
        print("❌ No medicine name found on packaging")
        return None

    top = candidates[0]
    print(f"✅ Packaging OCR complete — detected: '{top}'")
    return top


# ──────────────────────────────────────────────
# MODULE 4: Matcher with confidence levels
# ──────────────────────────────────────────────

def normalize(text):
    if not text:
        return ""
    text = text.upper()
    text = re.sub(r'[^\w\s-]', '', text)
    fillers = ['CAPSULE', 'CAPSULES', 'TABLET', 'TABLETS',
               'SYRUP', 'INJECTION', 'MG', 'ML', 'IU']
    for word in fillers:
        text = re.sub(rf'\b{word}\b', '', text)
    return ' '.join(text.split()).strip()


def get_confidence_level(score):
    """
    FIX RISK 2: Translate numeric score into human-readable confidence.
    Helps user understand HOW sure the system is.
    """
    if score >= 95:
        return "🟢 VERY HIGH"
    elif score >= 85:
        return "🟡 HIGH"
    elif score >= 75:
        return "🟠 MODERATE"
    else:
        return "🔴 LOW"


def verify_medicine(packaging_name, prescription_medicines, threshold=75):
    best_match = None
    best_score = 0

    for med in prescription_medicines:
        a = normalize(packaging_name)
        b = normalize(med['name'])
        if not a or not b:
            continue
        score = max(
            fuzz.ratio(a, b),
            fuzz.partial_ratio(a, b),
            fuzz.token_sort_ratio(a, b),
            fuzz.token_set_ratio(a, b)
        )
        if score > best_score:
            best_score = score
            best_match = med

    if best_score >= threshold:
        return best_match, best_score
    return None, best_score


# ──────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────

def run_pipeline(prescription_path, packaging_path):
    print("\n" + "=" * 60)
    print("🏥 MEDICINE VERIFICATION PIPELINE")
    print("=" * 60)

    # Step 1: OCR prescription
    raw_text = ocr_prescription(prescription_path)
    if not raw_text:
        print("❌ Pipeline stopped — prescription OCR failed")
        return

    # Step 2: Parse medicines
    print("\n[STEP 2] 🧹 Parsing prescription text...")
    medicines = parse_medicines(raw_text)
    if not medicines:
        print("❌ Pipeline stopped — no medicines parsed")
        return

    print(f"✅ Found {len(medicines)} medicines in prescription:")
    for i, m in enumerate(medicines, 1):
        status = "✅" if m['dosage'] != "Not found" else "⚠️ "
        print(f"   [{i}] {status} {m['name']} | Dosage: {m['dosage']}")

    # Step 3: OCR packaging
    detected_name = ocr_packaging(packaging_path)
    if not detected_name:
        print("❌ Pipeline stopped — packaging OCR failed")
        return

    # Step 4: Match and verify
    print("\n[STEP 4] 🔗 Matching and verifying...")
    matched, score = verify_medicine(detected_name, medicines)
    confidence = get_confidence_level(score)

    # Step 5: Final result
    print("\n" + "=" * 60)
    print("📋 FINAL VERIFICATION RESULT")
    print("=" * 60)
    print(f"📦 Scanned medicine  : {detected_name}")
    print(f"📊 Confidence        : {confidence} ({score}/100)")

    if matched:
        print(f"✅ STATUS            : CORRECT MEDICINE")
        print(f"💊 Matches           : {matched['name']}")
        print(f"💉 Dosage            : {matched['dosage']}")
        print(f"🕐 Timing            : {matched['timing']}")
        if matched['notes']:
            for note in matched['notes']:
                print(f"📝 Note              : {note}")

        # FIX RISK 1: Warn if confidence is only moderate
        if score < 85:
            print(f"\n⚠️  WARNING: Moderate confidence only ({score}/100)")
            print(f"   Please double-check medicine name manually")
    else:
        print(f"❌ STATUS            : WRONG MEDICINE")
        print(f"   '{detected_name}' is NOT in the prescription")
        print(f"   Best match score: {score}/100")
        print(f"\n⚠️  Do NOT administer this medicine without verification")

    print("=" * 60)


# ── Entry point ───────────────────────────────
if __name__ == "__main__":
    run_pipeline(
        prescription_path="images/prescriptions/prescription1.png",
        packaging_path="images/packaging/current_package.jpg"
    )