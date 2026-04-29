# check_setup.py
# Run this to confirm all libraries are installed correctly

import sys

def check_import(lib_name, import_name=None):
    try:
        __import__(import_name or lib_name)
        print(f"  ✅ {lib_name} — OK")
        return True
    except ImportError:
        print(f"  ❌ {lib_name} — NOT FOUND (run: pip install {lib_name})")
        return False

print("\n🔍 Checking Python libraries...")
check_import("opencv-python", "cv2")
check_import("pytesseract")
check_import("Pillow", "PIL")
check_import("fuzzywuzzy")
check_import("numpy")
check_import("ultralytics")

# Check Tesseract OCR engine
print("\n🔍 Checking Tesseract OCR engine...")
try:
    import pytesseract

    # ⚠️ Windows users: update this path to your Tesseract install location
    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    version = pytesseract.get_tesseract_version()
    print(f"  ✅ Tesseract — version {version}")
except Exception as e:
    print(f"  ❌ Tesseract — ERROR: {e}")
    print("     → Make sure Tesseract is installed and in your PATH")

# Check OpenCV camera access (optional at this stage)
print("\n🔍 Checking OpenCV version...")
import cv2
print(f"  ✅ OpenCV version: {cv2.__version__}")

print("\n✅ Setup check complete!\n")