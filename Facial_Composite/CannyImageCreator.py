import cv2
import numpy as np
from PIL import Image
from diffusers.utils import load_image

image = load_image("reference_images/my_face_3_18_768×1152_20_60_2510778634.png")

image = np.array(image)

low_threshold = 100
high_threshold = 200

image = cv2.Canny(image, low_threshold, high_threshold)
image = Image.fromarray(image)
image.save("reference_1_images/canny_my_face_3_18_768×1152_20_60_2510778634.png")