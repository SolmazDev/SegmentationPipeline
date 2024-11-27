import cv2
import os
import requests
import base64
import numpy as np
import time
import concurrent.futures
import argparse


def autoresize(image):
    """
    Resize the image if its height exceeds 1000 pixels.
    """
    height, width, _ = image.shape
    if height > 1000:
        factor = 0.5
        if height > 2000:
            factor = 0.25

        image = cv2.resize(image, (int(width * factor), int(height * factor)), interpolation=cv2.INTER_AREA)

    return image


def process_image(file_name, image, microservice_url, output_paths, retries=3):
    """
    Process a single image by resizing, sending to the microservice, and saving the outputs.
    """
    PATH2, PATH3 = output_paths

    for _ in range(retries):
        try:
            resized_image = autoresize(image)
            _, img_encoded = cv2.imencode('.jpg', resized_image)
            files = {"file": (file_name, img_encoded.tobytes(), "image/jpeg", {"Expires": "0"})}

            response = requests.post(microservice_url, files=files)
            if response.status_code == 200:
                response_data = response.json()
                decoded_mask = base64.b64decode(response_data['matte'])
                combined_mask = cv2.imdecode(np.frombuffer(decoded_mask, np.uint8), cv2.IMREAD_GRAYSCALE)

                # Save resized image and mask
                image_path = os.path.join(PATH2, file_name)
                mask_path = os.path.join(PATH3, file_name)
                cv2.imwrite(image_path, resized_image)
                cv2.imwrite(mask_path, combined_mask)

                print(f"Image {file_name} processed successfully.")
                return True
            else:
                print(f"Error processing image {file_name}. Status code: {response.status_code}")
                time.sleep(1)
        except Exception as e:
            print(f"Error processing image {file_name}: {e}")
            time.sleep(1)

    return False


def main():
    parser = argparse.ArgumentParser(description="Process images using a background segmentation microservice.")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to the folder containing input images.")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to the folder to save processed images.")
    parser.add_argument("--microservice_url", type=str, default="http://10.20.0.152:5660/u2net", help="URL of the microservice.")
    args = parser.parse_args()

    folder_path = args.input_folder
    output_path_resized = os.path.join(args.output_folder, "resized_images")
    output_path_masks = os.path.join(args.output_folder, "masks")

    os.makedirs(output_path_resized, exist_ok=True)
    os.makedirs(output_path_masks, exist_ok=True)

    output_paths = (output_path_resized, output_path_masks)

    image_files = [file_name for file_name in os.listdir(folder_path) if file_name.endswith(('.png', '.jpg', '.jpeg'))]

    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = []
        for file_name in image_files:
            image_path = os.path.join(folder_path, file_name)
            image = cv2.imread(image_path)
            futures.append(executor.submit(process_image, file_name, image, args.microservice_url, output_paths))

        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error in processing: {e}")

    end = time.time()
    print(f"Total processing time: {end - start:.2f} seconds")


if __name__ == "__main__":
    main()




