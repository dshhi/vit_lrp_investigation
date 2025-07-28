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
import os
import numpy as np
import matplotlib.pyplot as plt
from torchinfo import summary
from pathlib import Path
from scipy import stats
import torch.nn.functional as F

def calculate_distances_and_angles(saved_input, mlp_block):
    distances = []
    angles = []
    
    # Access the weights of the second linear layer (index 3)
    try:
        weights = None
        for name, param in mlp_block.named_parameters():
            if name == '3.weight':
                weights = param.data.cpu().numpy()
                break
        
        if weights is None:
            print("Second linear layer weights not found.")
            return []
    except Exception as e:
        print(f"Error accessing weights: {e}")
        return []

    # inputs shape: (197, 3072)
    # weights2 shape: (768, 3072)
    
    # Calculate distances to each weight vector
    for weight_vector in weights:
        # Reshape weight_vector to (1, 3072) to match input shape

        if isinstance(weight_vector, np.ndarray):
                weight_vector = torch.tensor(weight_vector)
        weight_vector = weight_vector.unsqueeze(0).to('cuda')  # Shape becomes (1, 3072)
        
        # Calculate Euclidean distances
        distance = torch.norm(saved_input - weight_vector, dim=1)  # Calculate distance for each input vector
        distances.append(distance)  # Store distances
        
        # Calculate angles using cosine similarity
        cosine_sim = F.cosine_similarity(saved_input, weight_vector, dim=1)  # Cosine similarity
        angle = torch.acos(cosine_sim)  # Calculate angle in radians
        angle_degrees = angle * (180 / np.pi)  # Convert to degrees
        angles.append(angle_degrees)  # Store angles

    return distances, angles

def calculate_angles(mlp_block):
    """
    Calculate angles between weight vectors in the MLP block.

    Parameters:
    mlp_block: The MLP block from which to extract weight vectors.

    Returns:
    angles: List of angles in degrees.
    """
    angles = []
    
    # Access the weights of the first linear layer (index 0)
    try:
        weights1 = mlp_block.named_parameters().__next__()[1].data.cpu().numpy()  # Get weights from the first linear layer
    except StopIteration:
        print("No parameters found in MLP block.")
        return []

    # Access the weights of the second linear layer (index 3)
    try:
        weights2 = None
        for name, param in mlp_block.named_parameters():
            if name == '3.weight':
                weights2 = param.data.cpu().numpy()
                break
        
        if weights2 is None:
            print("Second linear layer weights not found.")
            return []
    except Exception as e:
        print(f"Error accessing weights: {e}")
        return []

    # Combine weights from both layers
    all_weights = [weights2]
    
    angle_count = 0

    for weights in all_weights:
        for i in range(len(weights)):
            for j in range(i + 1, len(weights)):
                dot_product = np.dot(weights[i], weights[j])
                norm_i = np.linalg.norm(weights[i])
                norm_j = np.linalg.norm(weights[j])
                cos_theta = dot_product / (norm_i * norm_j)
                angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
                angles.append(angle)
                angle_count += 1  # Count the angles being computed

    print("Total angles computed:", angle_count)    
    return angles

saved_inputs = []

def create_input_hook():
    def input_hook(module, input, output):
        with torch.no_grad():
            # input shape: (batch, tokens, features)
            inputs = input[0].detach().clone()  # Assuming input is a tuple
            saved_inputs.append(inputs)  # Save the inputs directly

    return input_hook  # Return only the input_hook function

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


def plot_histogram(data, mlp_index, metric):
    """
    Create a histogram plot of the angles and save it.

    Parameters:
    data: List of data points to plot.
    mlp_index: Index of the MLP block for naming the file.
    """
    data = data.cpu().numpy()
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=30, color='blue', alpha=0.7)
    if metric == 'angle':
        plt.title(f'Angle Distribution of MLP Block {mlp_index}')
        plt.xlabel('Angle (degrees)')
    elif metric == 'distance':
        plt.title(f'Euclidian Distance Distribution of MLP Block {mlp_index}')
        plt.xlabel('Euclidian distance')
    plt.ylabel('Frequency')
    plt.grid(axis='y')

    # Calculate metrics
    mean = np.mean(data)
    median = np.median(data)
    mode = stats.mode(data)[0]
    std_dev = np.std(data)
    variance = np.var(data)
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    percentiles = np.percentile(data, [25, 50, 75])
    iqr = percentiles[2] - percentiles[0]
    data_range = np.max(data) - np.min(data)

    # Add metrics to plot
    metrics_text = (f'Mean: {mean:.2f}\n'
                    f'Median: {median:.2f}\n'
                    f'Mode: {mode:.2f}\n'
                    f'Std Dev: {std_dev:.2f}\n'
                    f'Variance: {variance:.2f}\n'
                    f'Skewness: {skewness:.2f}\n'
                    f'Kurtosis: {kurtosis:.2f}\n'
                    f'25th Percentile: {percentiles[0]:.2f}\n'
                    f'75th Percentile: {percentiles[2]:.2f}\n'
                    f'IQR: {iqr:.2f}\n'
                    f'Range: {data_range:.2f}')
    
    plt.text(0.7, 0.85, metrics_text, transform=plt.gca().transAxes, fontsize=10,
             bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout()

    # Create 'plots' directory if it doesn't exist
    Path('plots/input_superposition').mkdir(parents=True, exist_ok=True)
    if metric == 'angle':
        plt.savefig(f'plots/input_superposition/mlp_block_{mlp_index}_angles_histogram.png')
    elif metric == 'distance':
        plt.savefig(f'plots/input_superposition/mlp_block_{mlp_index}_distances_histogram.png')
    plt.close()

model, weights = get_vit_imagenet()
image = Image.open('cat_dog.jpg').convert('RGB')
image_resized = image.resize([224,224])
image_resized.save('input_resized.jpg')
image_resized.convert('L').save('input_resized_grayscale.jpg')
input_tensor = weights.transforms()(image).unsqueeze(0).to("cuda")

input_tensor.grad = None  # Reset gradients
summary(model, (input_tensor.shape))

mlp_blocks = []

# Access the encoder layers
for encoder_block in model.encoder.layers:
    # Each encoder block typically has an MLP layer named 'mlp'
    mlp_layer = encoder_block.mlp
    mlp_blocks.append(mlp_layer)


angles=[]
for index, mlp_block in enumerate(mlp_blocks):
    angles = calculate_angles(mlp_block)
    angles.append(angles)



for layer in model.encoder.layers:
    input_hook = create_input_hook()
    # Assuming model is defined
    layer.mlp[3].register_forward_hook(input_hook)

y = model(input_tensor.requires_grad_())

distances = []
angles = []
for index, mlp_block in enumerate(mlp_blocks):
    distances_mlp_block, angles_mlp_block = calculate_distances_and_angles(saved_inputs[index], mlp_block)
    distances.append(distances_mlp_block)
    angles.append(angles_mlp_block)

for mlp_block_index, mlp_block_distance in enumerate(distances):
    flattened_distances = torch.cat(mlp_block_distance, dim=0).view(-1)
    plot_histogram(flattened_distances, mlp_block_index, 'distance')
    flattened_angles = torch.cat(angles[mlp_block_index], dim=0).view(-1)
    plot_histogram(flattened_angles, mlp_block_index, 'angle')


pdb.set_trace()


for name, param in model.named_parameters():
        print(f"Parameter Name: {name}, Shape: {param.shape}")
