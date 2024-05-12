import torch
from torchvision import models, transforms
from PIL import Image, ImageDraw
import os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


def load_input_image(image_path):
    """Loads and changes picture to give it as inputs of SSD."""
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    original_size = image.size  # Capture original image size
    return image, transform(image).unsqueeze(0), original_size


def detect_objects(model, image, transformed_image, original_size, threshold=0.2):
    """Détection of the object thanks to a model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    transformed_image = transformed_image.to(device)
    with torch.no_grad():
        predictions = model(transformed_image)

    
    pred_scores = predictions[0]['scores']
    pred_boxes = predictions[0]['boxes']

    # Draw boxes
    draw = ImageDraw.Draw(image)
    for score, box in zip(pred_scores, pred_boxes):
        if score > threshold:
            box = [b.item() for b in box]
            # Convert back to original image dimensions
            box = [box[0] * original_size[0] / 300, box[1] * original_size[1] / 300,
                   box[2] * original_size[0] / 300, box[3] * original_size[1] / 300]
            draw.rectangle(box, outline='red', width=5)
            print(f"Score: {score.item():.2f}, Box: {box}")
    return image


model = models.detection.ssd300_vgg16(pretrained=True)
model.eval()

folder_path = r"C:\Users\zakar\Documents\Cours\Projets\Projet MA1\banana"
for image_file in os.listdir(folder_path):
    image_path = os.path.join(folder_path, image_file)
    if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        print(f"Détection dans l'image: {image_path}")
        original_image, transformed_image, original_size = load_input_image(image_path)
        result_image = detect_objects(model, original_image, transformed_image, original_size)
        plt.figure()
        plt.imshow(result_image)
        plt.axis('off')
        plt.show()
