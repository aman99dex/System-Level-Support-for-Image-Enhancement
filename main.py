import rawpy
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Define the file path for the RAW image
file_path = "C:/Users/kumar/Desktop/python/RAW_SONY_ILCE-7RM2.ARW"

# Step 1: Load RAW image with error handling
def load_raw_image(file_path):
    try:
        with rawpy.imread(file_path) as raw:
            rgb_image = raw.postprocess()
        return rgb_image
    except rawpy.LibRawIOError as e:
        print(f"Error loading RAW file: {e}")
        return None

# Step 2: Apply white balance correction
def correct_white_balance(image):
    result = cv2.xphoto.createSimpleWB().balanceWhite(image)
    return result

# Step 3: Resize image
def resize_image(image, target_size=(2000, 2000)):
    resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    return resized_image

# Step 4: Apply CLAHE for color correction
def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final

# Step 5: Selective sharpening
def selective_sharpen(image, focus_rect):
    x1, y1, x2, y2 = focus_rect
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    blurred = cv2.GaussianBlur(image, (0, 0), 3)
    sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
    return cv2.bitwise_and(sharpened, sharpened, mask=mask) + cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask))

# Step 6: Enhance saturation
def enhance_saturation(image, saturation_scale=1.5):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.multiply(s, saturation_scale)
    s = np.clip(s, 0, 255).astype(hsv.dtype)
    enhanced_hsv = cv2.merge([h, s, v])
    enhanced_image = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)
    return enhanced_image

# Main processing function
def process_image(file_path):
    rgb_image = load_raw_image(file_path)
    if rgb_image is None:
        print("Image loading failed. Exiting process.")
        return None

    # Assuming we detected the cat's region, let's hardcode an example bounding box for illustration
    focus_rect = (500, 500, 1500, 1500)  # x1, y1, x2, y2 - Adjust these values for actual cat location

    with ThreadPoolExecutor() as executor:
        future_balance = executor.submit(correct_white_balance, rgb_image)
        future_resize = executor.submit(resize_image, rgb_image)
        
        balanced_image = future_balance.result()
        resized_image = future_resize.result()

        color_corrected_image = apply_clahe(resized_image)

        future_sharpen = executor.submit(selective_sharpen, color_corrected_image, focus_rect)
        future_saturation = executor.submit(enhance_saturation, color_corrected_image)

        sharpened_image = future_sharpen.result()
        saturated_image = future_saturation.result()

    final_image = cv2.addWeighted(sharpened_image, 0.7, saturated_image, 0.3, 0)
    return final_image

# Function to resize image to fit the display
def resize_to_display(image):
    screen_res = 1280, 720  # Example screen resolution, adjust as needed
    scale_width = screen_res[0] / image.shape[1]
    scale_height = screen_res[1] / image.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(image.shape[1] * scale)
    window_height = int(image.shape[0] * scale)
    return cv2.resize(image, (window_width, window_height))

# Function to display the final image
def display_image(image):
    if image is not None:
        resized_image = resize_to_display(image)
        cv2.imshow('Enhanced Image (2K Resolution, Cat Focus)', resized_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No image to display.")

if __name__ == "__main__":
    final_image = process_image(file_path)
    display_image(final_image)
