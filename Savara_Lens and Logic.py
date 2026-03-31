import cv2
import numpy as np
import os
import glob

# This finds the folder where THIS script is saved (your Desktop)
script_dir = os.path.dirname(os.path.abspath(__file__))


def find_file(pattern):
    """Finds a file matching the pattern even if extension is weird."""
    files = glob.glob(os.path.join(script_dir, f"*{pattern}*"))
    return files[0] if files else None


def process(input_pattern, output_name, mode):
    img_path = find_file(input_pattern)
    if not img_path:
        print(
            f"❌ Error: Could not find any file containing '{input_pattern}' in {script_dir}")
        return

    img = cv2.imread(img_path)
    if img is None:
        print(
            f"❌ Error: OpenCV couldn't read {img_path}. Check if it's open in another app.")
        return

    print(f"Found: {os.path.basename(img_path)}. Processing...")

    if mode == 'past':
        # Sepia + Grain + Vignette
        kernel = np.array(
            [[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]])
        res = np.clip(cv2.transform(img, kernel), 0, 255).astype(np.uint8)
        noise = np.random.normal(0, 15, res.shape).astype(np.int16)
        res = np.clip(cv2.add(res.astype(np.int16), noise),
                      0, 255).astype(np.uint8)
        r, c = res.shape[:2]
        mask = (cv2.getGaussianKernel(r, r/2.5) *
                cv2.getGaussianKernel(c, c/2.5).T)
        mask = mask / mask.max()
        for i in range(3):
            res[:, :, i] = res[:, :, i] * mask
    else:
        # Future: Darken + Neon
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = cv2.multiply(hsv[:, :, 2], 0.3)  # Darken
        dark = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        edges = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(
            img, cv2.COLOR_BGR2GRAY), (5, 5), 0), 50, 150)
        neon = cv2.cvtColor(cv2.dilate(edges, np.ones(
            (3, 3), np.uint8)), cv2.COLOR_GRAY2BGR)
        neon[np.where((neon > [0, 0, 0]).all(axis=2))] = [255, 255, 0]
        res = cv2.addWeighted(
            dark, 1.0, cv2.GaussianBlur(neon, (15, 15), 0), 1.2, 0)

    cv2.imwrite(os.path.join(script_dir, output_name), res)
    print(f"✅ Success! Saved to {output_name}")


if __name__ == "__main__":
    # We use partial names (image_2cd31c) to avoid extension errors
    process("2cd31c", "past_themed_output.jpg", "past")
    process("2cd2e2", "future_themed_output.jpg", "future")
