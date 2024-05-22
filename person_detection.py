import torch
from PIL import Image, ImageDraw
import requests
import time
import cv2
from transformers import AutoImageProcessor, AutoModelForObjectDetection

processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = AutoModelForObjectDetection.from_pretrained("facebook/detr-resnet-50")

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

VISUALIZE = True

while True:
    time.sleep(5)
    vc = cv2.VideoCapture(0)

    if vc.isOpened():  # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False
        frame = None

    if rval:
        # image = Image.open("/Users/victorsahin/Desktop/Screenshot 2024-05-22 at 12.33.48 PM.png").convert("RGB")
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.95)[0]

        score, label, box = sorted([e for e in zip(results["scores"], results["labels"], results["boxes"])
                                    if model.config.id2label[e[1].item()] == "person"], key=lambda x: -x[0])[0]
        box = [round(i, 2) for i in box.tolist()]
        if model.config.id2label[label.item()] == "person":
            print(
                f"Detected {model.config.id2label[label.item()]} with confidence "
                f"{round(score.item(), 3)} at location {box}"
            )
            if VISUALIZE:
                draw = ImageDraw.Draw(image)
                draw.rectangle((tuple(box[0:2]), tuple(box[2:])))
                image.show()
        midpoint = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

        print("=================================")
