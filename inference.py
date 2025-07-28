import torch
import itertools
from PIL import Image
from torchvision.models import vision_transformer

from zennit.image import imgify
from zennit.composites import LayerMapComposite
import zennit.rules as z_rules

from lxt.efficient import monkey_patch, monkey_patch_zennit
import pdb
import PIL
import torchvision.transforms as transforms

monkey_patch(vision_transformer, verbose=True)
monkey_patch_zennit(verbose=True)

def get_vit_imagenet(device="cuda"):
    """
    Load a pre-trained Vision Transformer (ViT) model with ImageNet weights.

    Parameters:
    device (str): Device to load the model on ('cuda' or 'cpu')

    Returns:
    tuple: (model, weights) - The ViT model and its pre-trained weights
    """
    weights =vision_transformer.ViT_B_16_Weights.IMAGENET1K_V1
    model = vision_transformer.vit_b_16(weights=weights)
    # weights =vision_transformer.ViT_L_16_Weights.IMAGENET1K_V1
    # model = vision_transformer.vit_l_16(weights=weights)
    model.eval()
    model.to(device)

    # Deactivate gradients on parameters to save memory
    for param in model.parameters():
        param.requires_grad = False

    return model, weights

# Load the pre-trained ViT model
model, weights = get_vit_imagenet()

def create_hook(topk=100):
    def forward_hook(module, input, output):
        with torch.no_grad():
            # output shape: (batch, tokens, features)
            # Flatten activations per token, keep only top 100 per token
            activ = output.detach().clone()
            # activ shape: (batch, tokens, features)
            batch, tokens, features = activ.shape
            values, indices = torch.topk(activ.abs(), topk, dim=2)
            mask = torch.zeros_like(activ)
            mask.scatter_(2, indices, 1)
        return output * mask
    return forward_hook

# for layer in model.encoder.layers:
#     layer.mlp[3].register_forward_hook(create_hook(100))
#     pass

for layer in model.encoder.layers[-3:]:
    layer.mlp[3].register_forward_hook(create_hook(1))
    pass

# Load and preprocess the input image
image = Image.open('cat_dog.jpg').convert('RGB')
image_resized = image.resize([224,224])
image_resized.save('input_resized.jpg')
image_resized.convert('L').save('input_resized_grayscale.jpg')
input_tensor = weights.transforms()(image).unsqueeze(0).to("cuda")
img_size = image.size  # (width, height)

# Store the generated heatmaps
heatmaps = []

#############################################################
# Use topk activations for experimenting with epsilon rule:
#############################################################

input_tensor.grad = None  # Reset gradients
zennit_comp = LayerMapComposite([
    (torch.nn.Conv2d, z_rules.ZPlus()),
    (torch.nn.Linear, z_rules.Epsilon()),
])
# zennit_comp = LayerMapComposite([
#     (torch.nn.Conv2d, z_rules.Epsilon()),
#     (torch.nn.Linear, z_rules.Epsilon()),
# ])

# Register the composite rules with the model
zennit_comp.register(model)

# Forward pass with gradient tracking enabled
y = model(input_tensor.requires_grad_())

# Get the top 5 predictions
_, top5_classes = torch.topk(y, 5, dim=1)
top5_classes = top5_classes.squeeze(0).tolist()

# Get the class labels
labels = weights.meta["categories"]
top5_labels = [labels[class_idx] for class_idx in top5_classes]

# Print the top 5 predictions and their labels
for i, class_idx in enumerate(top5_classes):
    print(f'Top {i+1} predicted class: {class_idx}, label: {top5_labels[i]}')

# Backward pass for the highest probability class
# This initiates the LRP computation through the network
y[0, 156].backward()

# Remove the registered composite to prevent interference in future iterations
zennit_comp.remove()

# Calculate the relevance by computing Gradient * Input
# This is the final step of LRP to get the pixel-wise explanation
heatmap = (input_tensor * input_tensor.grad).sum(1)

# Normalize relevance between [-1, 1] for plotting
heatmap = heatmap / abs(heatmap).max()
heatmap = heatmap.detach().cpu().numpy()

img = imgify(heatmap, vmin=-1, vmax=1)
img.convert('RGB').save('vit_heatmap.jpg')
img_resized = img.resize(img_size, PIL.Image.Resampling.LANCZOS).convert('RGB')

# Save the resized image with a better quality
img_resized.save('vit_heatmap_orig_size.jpg', quality=100)  #



#############################################################
# Experiment with different gamma values for Conv2d and Linear layers
# Gamma is a hyperparameter in LRP that controls how much positive vs. negative
# contributions are considered in the explanation
#############################################################
input_tensor.grad = None  # Reset gradients

# Define rules for the Conv2d and Linear layers using 'zennit'
# LayerMapComposite maps specific layer types to specific LRP rule implementations
conv_gamma = 100
lin_gamma = 1
print("Gamma Conv2d:", conv_gamma, "Gamma Linear:", lin_gamma)
zennit_comp = LayerMapComposite([
    (torch.nn.Conv2d, z_rules.Gamma(conv_gamma)),
    (torch.nn.Linear, z_rules.Gamma(lin_gamma)),
])

# Register the composite rules with the model
zennit_comp.register(model)

# Forward pass with gradient tracking enabled
y = model(input_tensor.requires_grad_())

# Get the top 5 predictions
_, top5_classes = torch.topk(y, 5, dim=1)
top5_classes = top5_classes.squeeze(0).tolist()

# Get the class labels
labels = weights.meta["categories"]
top5_labels = [labels[class_idx] for class_idx in top5_classes]

# Print the top 5 predictions and their labels
for i, class_idx in enumerate(top5_classes):
    print(f'Top {i+1} predicted class: {class_idx}, label: {top5_labels[i]}')

# Backward pass for the highest probability class
# This initiates the LRP computation through the network
y[0, 156].backward()

# Remove the registered composite to prevent interference in future iterations
zennit_comp.remove()

# Calculate the relevance by computing Gradient * Input
# This is the final step of LRP to get the pixel-wise explanation
heatmap_gamma = (input_tensor * input_tensor.grad).sum(1)

# Normalize relevance between [-1, 1] for plotting
heatmap_gamma = heatmap_gamma / abs(heatmap_gamma).max()

heatmap_gamma = heatmap_gamma.detach().cpu().numpy()

# Visualize all heatmaps in a grid (3×5) and save to a file
# vmin and vmax control the color mapping range
#imgify(heatmaps, vmin=-1, vmax=1).save('vit_heatmap.png')
img_gamma = imgify(heatmap_gamma, vmin=-1, vmax=1)
img_gamma.convert('RGB').save('vit_heatmap_gamma.jpg')
#imgify(heatmaps, vmin=-1, vmax=1).save('vit_heatmap.png')
img_gamma_resized = img_gamma.resize(img_size, PIL.Image.Resampling.LANCZOS).convert('RGB')



# Save the resized image with a better quality
img_gamma_resized.save('vit_heatmap_gamma_orig_size.jpg', quality=100)  #
#imgify(heatmaps[0], vmin=-1, vmax=1).save('vit_heatmap.png')
