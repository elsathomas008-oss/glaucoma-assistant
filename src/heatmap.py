import cv2
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image

class GlaucomaGradCAM:
    def __init__(self, model_path=None):
        # Using a ResNet50 backbone for classification
        self.model = models.resnet50(pretrained=True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 2)
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()

        self.gradients = None
        self.activations = None
        self.model.layer4.register_forward_hook(self._save_activation)
        self.model.layer4.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def process_and_generate(self, image_path, save_output_path="outputs/heatmap.jpg"):
        img = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        input_tensor = transform(img).unsqueeze(0)

        output = self.model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        glaucoma_score = probabilities[0][1].item() * 100

        # Compute Grad-CAM
        self.model.zero_grad()
        output[0][1].backward()

        pooled_grads = torch.mean(self.gradients, dim=[0, 2, 3])
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_grads[i]

        heatmap = torch.mean(self.activations, dim=1).squeeze().detach().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= (np.max(heatmap) + 1e-8)

        # Overlay heatmap onto original fundus photo
        orig_img = cv2.imread(image_path)
        orig_img = cv2.resize(orig_img, (512, 512))
        heatmap_resized = cv2.resize(heatmap, (512, 512))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

        superimposed = cv2.addWeighted(orig_img, 0.6, heatmap_colored, 0.4, 0)
        cv2.imwrite(save_output_path, superimposed)

        return glaucoma_score, save_output_path