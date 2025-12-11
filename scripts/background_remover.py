# encode:utf-8

import cv2
import os

# ================= Script parameters =================

INPUT_FOLDER: str = "src/3dgs_pipeline/dataset/1"
OUTPUT_FOLDER: str = "src/3dgs_pipeline/dataset/1/input_no_bg"
THRESHOLD: int = 80


# ===================================================


def get_images_path(path: str) -> list[str]:
    """
    Get all image file paths from the specified directory.

    Args:
        path (str): The directory path to search for images.
    Returns:
        list[str]: A list of image file paths.
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_paths = []
    for root, _, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[1].lower() in image_extensions:
                image_paths.append(os.path.join(root, file))
    return image_paths


def remove_dark_areas(source_image_path: str, mask_image_path: str, output_path: str, value: int) -> None:
    """
    Remove areas in the source image based on dark areas in the mask image.

    Args:
        source_image_path (str): Path to the source image.
        mask_image_path (str): Path to the mask image.
        output_path (str): Path to save the output image.
        value (int): Threshold value to determine dark areas.
    """
    source_image = cv2.imread(source_image_path)
    mask_image = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)
    if source_image.shape[:2] != mask_image.shape[:2]:
        print(f"Size mismatch between source and mask images: {source_image_path} and {mask_image_path}")
        return
    _, binary_mask = cv2.threshold(mask_image, value, 255, cv2.THRESH_BINARY)
    binary_mask_3ch = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
    result_image = cv2.bitwise_and(source_image, binary_mask_3ch)
    cv2.imwrite(output_path, result_image)


def process() -> None:
    """
    Process all images in the input folder to remove dark areas based on corresponding mask images.
    """
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    source_images = get_images_path(INPUT_FOLDER + "/input")
    for source_image_path in source_images:
        image_name = os.path.basename(source_image_path)
        mask_image_path = os.path.join(INPUT_FOLDER, "depth", image_name)
        output_image_path = os.path.join(OUTPUT_FOLDER, image_name)
        if os.path.exists(mask_image_path):
            remove_dark_areas(source_image_path, mask_image_path, output_image_path, value=THRESHOLD)
        else:
            print(f"Mask image not found for {source_image_path}")


if __name__ == "__main__":
    process()
