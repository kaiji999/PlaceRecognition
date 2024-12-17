import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# Load the Places365 pretrained AlexNet model
model = models.alexnet(pretrained=False)
model.classifier[6] = torch.nn.Linear(4096, 365)  # Adjust the output layer for 365 classes

# Load the weights from the .tar file
checkpoint = torch.hub.load_state_dict_from_url(
    'http://places2.csail.mit.edu/models_places365/alexnet_places365.pth.tar',
    map_location=torch.device('cpu')
)
state_dict = checkpoint['state_dict']  # Extract the actual weights
state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}  # Remove 'module.' prefix
model.load_state_dict(state_dict)

model.eval()  # Set the model to evaluation mode

# Define preprocessing steps
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load class labels
with open('categories_places365.txt', 'r') as f:
    classes = [line.strip().split(' ')[0][3:] for line in f]

def load_image(image_path):
    """
    Load and preprocess the image.
    """
    image = Image.open(image_path).convert('RGB')  # Ensure 3 color channels
    return preprocess(image).unsqueeze(0)  # Add batch dimension

def predict_place(image_path):
    """
    Predict the place category for a given image.
    """
    # Load and preprocess the image
    image_tensor = load_image(image_path)
    
    # Perform inference
    outputs = model(image_tensor)
    _, predicted_idx = outputs.max(1)
    
    # Return the predicted class
    return classes[predicted_idx.item()]

if __name__ == '__main__':
    # Replace 'example.jpg' with your image file path
    image_path = 'example.jpg'
    
    # Make a prediction
    try:
        predicted_place = predict_place(image_path)
        print(f"Predicted Place: {predicted_place}")

        # Display the image and prediction
        image = Image.open(image_path)
        plt.imshow(image)
        plt.title(f"Predicted: {predicted_place}")
        plt.axis('off')
        plt.show()
    except FileNotFoundError:
        print(f"Error: The file '{image_path}' was not found. Please check the path and try again.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
