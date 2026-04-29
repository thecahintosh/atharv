# capture_medicine.py
# Goal: Load or capture a medicine packaging image and prepare it

import cv2
import os


def load_from_file(image_path):
    """Load a medicine package image from a saved file."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not load image: {image_path}")
        return None
    print(f"✅ Image loaded from file — {img.shape[1]}x{img.shape[0]} pixels")
    return img


def capture_from_camera(save_path="images/packaging/captured.jpg"):
    """
    Open webcam, show live preview, capture on keypress.
    Press SPACE to capture, Q to quit.
    """
    print("\n📷 Opening camera...")
    print("   → Position the medicine package clearly in frame")
    print("   → Make sure text on packaging is visible and well-lit")
    print("   → Press SPACE to capture")
    print("   → Press Q to quit\n")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Could not open camera.")
        print("   → Check if webcam is connected")
        print("   → Try changing VideoCapture(0) to VideoCapture(1)")
        return None

    captured_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read from camera.")
            break

        # Show live preview with instructions overlay
        display = frame.copy()
        cv2.putText(display, "SPACE=Capture  Q=Quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)
        cv2.putText(display, "Position medicine package in frame",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 1)

        cv2.imshow("Medicine Package Capture", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == ord('Q'):
            print("❌ Capture cancelled.")
            break

        elif key == ord(' '):
            captured_frame = frame.copy()
            print("✅ Image captured!")

            # Save captured image
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, captured_frame)
            print(f"✅ Saved to: {save_path}")
            break

    cap.release()
    cv2.destroyAllWindows()
    return captured_frame


def validate_image(img):
    """
    Basic checks — is the image usable for OCR?
    Warns about common problems: too dark, too blurry.
    """
    if img is None:
        return False

    import numpy as np

    # Check 1: Brightness (mean pixel value)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = gray.mean()

    if brightness < 50:
        print("⚠️  Warning: Image is very dark — try better lighting")
    elif brightness > 220:
        print("⚠️  Warning: Image is overexposed — reduce light or move back")
    else:
        print(f"✅ Brightness OK ({brightness:.1f}/255)")

    # Check 2: Blurriness (Laplacian variance — higher = sharper)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    if blur_score < 50:
        print(f"⚠️  Warning: Image may be blurry (score: {blur_score:.1f}) — hold camera steady")
    else:
        print(f"✅ Sharpness OK (score: {blur_score:.1f})")

    return True


def get_medicine_image(source="file",
                       file_path="images/packaging/medicine1.jpg"):
    """
    Main function — get image from file or camera.
    source = 'file' or 'camera'
    """
    print(f"\n📦 Getting medicine package image (source: {source})")
    print("-" * 50)

    if source == "camera":
        img = capture_from_camera()
    else:
        img = load_from_file(file_path)

    if img is None:
        return None

    # Validate quality
    validate_image(img)

    # Save a copy for pipeline use
    output_path = "images/packaging/current_package.jpg"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)
    print(f"✅ Working copy saved → {output_path}")

    return img


# ── Main ──────────────────────────────────────────────────
if __name__ == "__main__":

    # ── Option A: Load from file (use this for now) ──
    img = get_medicine_image(
        source="file",
        file_path="images/packaging/medicine.png"
    )

    # ── Option B: Capture from camera (uncomment when ready) ──
    # img = get_medicine_image(source="camera")

    if img is not None:
        print("\n✅ Image ready for next steps!")
        cv2.imshow("Medicine Package", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()