import cv2

image_path = "reference_images/proper_llm_my_face_2_0_1024_20_60_0.png"

img = cv2.imread(image_path)

size = 1024
new_size = 128
start_x = 340 + 10
start_y = 450 - 20
end_x = start_x + new_size
end_y = start_y + new_size

roi = img[start_y:end_y, start_x:end_x]
resize_roi = cv2.resize(roi, (size, size), interpolation = cv2.INTER_LANCZOS4)
cv2.imshow("image", resize_roi)
cv2.waitKey(0)

filename = 'crop_test_eye_my_face_3.png'
cv2.imwrite(filename, resize_roi)