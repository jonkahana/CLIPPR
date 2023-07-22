import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from model import *
from dataset import PROMPTS
# Define the transformation for the input image
preprocess = transforms.Compose([transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                      std=[0.229, 0.224, 0.225])])

# Load the discriminative model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLIP_Visual(classes=None, device=device).to(device)
model.load_state_dict(torch.load('weights/utk/clippr__gaussian/epoch_best.pth'))
model.eval()

classes = ['1', '10', '100+', '11', '12', '13', '14', '15', '16', '17', '18',
           '19', '2', '20', '21', '22', '23', '24', '25', '26', '27', '28',
           '29', '3', '30', '31', '32', '33', '34', '35', '36', '37', '38',
           '39', '4', '40', '41', '42', '43', '44', '45', '46', '47', '48',
           '49', '5', '50', '51', '52', '53', '54', '55', '56', '57', '58',
           '59', '6', '60', '61', '62', '63', '64', '65', '66', '67', '68',
           '69', '7', '70', '71', '72', '73', '74', '75', '76', '77', '78',
           '79', '8', '80', '81', '82', '83', '84', '85', '86', '87', '88',
           '89', '9', '90', '91', '92', '93', '95', '96', '99']
zero_shot_model = CLIP_Zero_Shot(classes=classes, prompt=PROMPTS['utk']).to(device)
zero_shot_model.eval()


# Define the prediction function
def predict(image):
    global model, zero_shot_model, preprocess, device
    # Preprocess the image
    image = Image.fromarray(image.astype('uint8'), 'RGB')
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    # Use GPU if available
    input_batch = input_batch.to(device)
    model = model.to(device)
    zero_shot_model = zero_shot_model.to(device)

    # Make the prediction
    with torch.no_grad():
        clippr_pred = int(np.round((model(input_batch)[0].item())))
        clip_pred = zero_shot_model(input_batch).argmax(dim=1, keepdim=True)[0].item()

    return clippr_pred, clip_pred


# Define the input and output interfaces
inputs = gr.inputs.Image()
outputs = [gr.outputs.Textbox(label="CLIPPR"),
           gr.outputs.Textbox(label="CLIP")]

# Create the Gradio interface
gr.Interface(fn=predict, inputs=inputs, outputs=outputs, title="Image Classifier").launch()
