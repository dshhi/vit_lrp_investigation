import cv2
import numpy as np
import pdb
from PIL import Image
from skimage.metrics import structural_similarity



def calculate_noise_metrics(ref_img_path, img_path):
    # Load the image
    ref_image = cv2.imread(ref_img_path)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the grayscale image
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the noise by subtracting the blurred image from the original grayscale image
    noise = gray_image - image

    # Normalize the noise to the range [0, 255] for visualization
    noise_normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

    # Convert the noise to a PIL image
    noise_image = Image.fromarray(noise_normalized.astype(np.uint8))
    # Optionally, save the noise image
    noise_image.save('noise.png')


    # Calculate the mean and standard deviation of the noise
    mean_noise = np.mean(noise)
    std_noise = np.std(noise)
    
    (score, diff) = structural_similarity(gray_image, image, full=True)
    print("Image similarity", score)

    return mean_noise, std_noise

ref_img_path = "vit_heatmap_gamma.jpg"
img_path = "vit_heatmap_no_hook.png"
mean_noise,std_noise = calculate_noise_metrics(ref_img_path,img_path)
print(f"mean_noise: {mean_noise}, std_noise: {std_noise}")
