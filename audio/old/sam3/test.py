import cv2
import torch
#################################### For Image ####################################
from PIL import Image
from sam3.model_builder import build_sam3_image_model
# from sam3.model.sam3_image_processor import Sam3Processor
# # Load the model
# model = build_sam3_image_model()
# processor = Sam3Processor(model)
print("Model loaded successfully")
cap = cv2.VideoCapture(0)

while True:
    # print("AAA")
    ret, frame = cap.read()
    if not ret:
        break
    # print("wtf")
    cv2.imshow("camera", frame)
    # print("hein")
    # break
    #
    # inference_state = processor.set_image(frame)
    # break

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
