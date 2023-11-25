from PIL import Image
import os

def get_image_info(image_path):
    try:
        with Image.open(image_path) as img:
            resolution = img.info.get("dpi", (0, 0))
            size = os.path.getsize(image_path)  # in bytes
            return {
                'filename': os.path.basename(image_path),
                'resolution': resolution,
                'width': img.width,
                'height': img.height,
                'size': size
            }
    except Exception as e:
        return {
            'filename': os.path.basename(image_path),
            'error': str(e)
        }

def check_image_dimensions(folder_path, target_width=1920, target_height=1080):
    image_info_list = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            image_path = os.path.join(folder_path, filename)
            image_info = get_image_info(image_path)
            image_info_list.append(image_info)

    # Check if all images have the specified dimensions
    for image_info in image_info_list:
        if 'error' not in image_info:
            if image_info['width'] != target_width or image_info['height'] != target_height:
                return False  # Image dimensions don't match the target dimensions

    return True  # All images have the specified dimensions

if __name__ == "__main__":
    folder_path = '/Users/divyamyadav/Desktop/Learning/ml/env/learning_microstructure_evolution/data/test/dataset'  # Change this to the path of your image folder
    target_width = 1920
    target_height = 1080

    if check_image_dimensions(folder_path, target_width, target_height):
        print("All images have the specified dimensions.")
    else:
        print("Not all images have the specified dimensions.")
