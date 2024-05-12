# Faster CNN

import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Load model 
model = fasterrcnn_resnet50_fpn(weights=True)
model.eval()  


# Transform pictures to have a good compatibility
def transform_image(image):
    transform = T.Compose([
        T.ToTensor(),  
    ])
    return transform(image).unsqueeze(0)  



def detect_objects(model, image_path):
    image = Image.open(image_path)  # Image loading
    image_tensor = transform_image(image)

    with torch.no_grad():
        predictions = model(image_tensor)  # Prédiction of object

    return image, predictions


# Path to the picture
image_path = r"C:\Users\zakar\Documents\Cours\Projets\Projet MA1\scissors\Scissors_3.jpg"

# Object detection
image, predictions = detect_objects(model, image_path)

# Print the results
scores = predictions[0]['scores'].numpy()
boxes = predictions[0]['boxes'].detach().numpy()
labels = predictions[0]['labels'].numpy()


score_threshold = 0.8

draw = ImageDraw.Draw(image)
for score, box, label in zip(scores, boxes, labels):
    if score > score_threshold:
        x1, y1, x2, y2 = box
        name = str(label)
        draw.rectangle([(x1, y1), (x2, y2)], outline='red')
        draw.text((x1, y1), name)

plt.imshow(image)
plt.axis('off')
plt.show()

print(scores)
