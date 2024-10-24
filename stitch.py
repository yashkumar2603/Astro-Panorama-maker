from pyimagesearch.panorama import Stitcher
import cv2
import os

# Function to automatically load images in the format image1, image2, ..., imagen
def load_images_from_folder(folder):
    images = []
    i = 1
    while True:
        image_path = os.path.join(folder, f"image{i}.jpg")
        if not os.path.exists(image_path):
            break  # Stop when no more images in the format image{i}.jpg are found
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Unable to load {image_path}")
            break
        images.append(cv2.resize(image, (400, int(image.shape[0] * 400 / image.shape[1]))))  # Resize to width 400
        i += 1
    return images

# Main function
def main():
    # Load images from the "images" folder
    folder = "images"  # Path to the folder containing the images
    images = load_images_from_folder(folder)

    if len(images) < 2:
        print("Not enough images to stitch. Exiting.")
        return

    # Initialize the stitcher
    stitcher = Stitcher()

    # Start stitching with the first image and stitch subsequent images one by one
    result = images[0]
    for i in range(1, len(images)):
        print(f"Stitching image {i+1} of {len(images)}...")
        result = stitcher.stitch([result, images[i]])

        if result is None:
            print(f"Image stitching failed at image {i+1}. Exiting.")
            return

    # Display the final stitched panorama
    cv2.imshow("Stitched Panorama", result)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
