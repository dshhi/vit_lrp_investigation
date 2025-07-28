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
import argparse

monkey_patch(vision_transformer, verbose=True)
monkey_patch_zennit(verbose=True)


# Initialize the argument parser
parser = argparse.ArgumentParser(description='Select model type for Vision Transformer.')
parser.add_argument('--model', type=str, choices=['vitb16', 'vitl16'], required=True,
                    help='Specify the model to use: vitb16 or vitl16')


# Load the pre-trained ViT model based on the argument
def get_vit_imagenet(device="cuda"):
    """
    Load a pre-trained Vision Transformer (ViT) model with ImageNet weights.
    """
    if args.model == 'vitb16':
        weights = vision_transformer.ViT_B_16_Weights.IMAGENET1K_V1
        model = vision_transformer.vit_b_16(weights=weights)
    elif args.model == 'vitl16':
        weights = vision_transformer.ViT_L_16_Weights.IMAGENET1K_V1
        model = vision_transformer.vit_l_16(weights=weights)

    model.eval()
    model.to(device)

    # Deactivate gradients on parameters to save memory
    for param in model.parameters():
        param.requires_grad = False

    return model, weights


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
    # try:
    #     weights1 = mlp_block.named_parameters().__next__()[1].data.cpu().numpy()  # Get weights from the first linear layer
    # except StopIteration:
    #     print("No parameters found in MLP block.")
    #     return []

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
    
    # Calculate angles between all pairs of weights
    for weights in all_weights:
        for i in range(len(weights)):
            for j in range(i + 1, len(weights)):
                dot_product = np.dot(weights[i], weights[j])
                norm_i = np.linalg.norm(weights[i])
                norm_j = np.linalg.norm(weights[j])
                cos_theta = dot_product / (norm_i * norm_j)
                angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
                angles.append(angle)
    
    return angles


def calculate_metrics(angles):
    """
    Calculate various metrics from the angles.

    Parameters:
    angles: List of angles to calculate metrics from.

    Returns:
    A dictionary containing the calculated metrics.
    """
    mean = np.mean(angles)
    median = np.median(angles)
    mode = stats.mode(angles)[0]
    std_dev = np.std(angles)
    variance = np.var(angles)
    skewness = stats.skew(angles)
    kurtosis = stats.kurtosis(angles)
    percentiles = np.percentile(angles, [25, 50, 75])
    iqr = percentiles[2] - percentiles[0]
    data_range = np.max(angles) - np.min(angles)
    min_angle = np.min(angles)  # Minimum angle
    max_angle = np.max(angles)  # Maximum angle

    return {
        'Mean': mean,
        'Median': median,
        'Mode': mode,
        'Standard Deviation': std_dev,
        'Variance': variance,
        'Skewness': skewness,
        'Kurtosis': kurtosis,
        '25th Percentile': percentiles[0],
        '75th Percentile': percentiles[2],
        'IQR': iqr,
        'Range': data_range,
        'Min Angle': min_angle,
        'Max Angle': max_angle
    }


def save_metrics_to_markdown_table(all_metrics):
    """
    Save all calculated metrics to a markdown file as a table.

    Parameters:
    all_metrics: List of dictionaries containing metrics for all MLP blocks.
    """
    # Prepare the header
    header = "| MLP Block | Mean | Median | Mode | Std Dev | Variance | Skewness | Kurtosis | 25th Percentile | 75th Percentile | IQR | Range | Min Angle | Max Angle |\n"
    separator = "|-----------|------|--------|------|---------|----------|----------|----------|------------------|------------------|-----|-------|-----------|-----------|\n"
    
    # Prepare the rows
    rows = ""
    for index, metrics in enumerate(all_metrics):
        row = (f"| {index} | {metrics['Mean']:.2f} | {metrics['Median']:.2f} | "
               f"{metrics['Mode']:.2f} | {metrics['Standard Deviation']:.2f} | "
               f"{metrics['Variance']:.2f} | {metrics['Skewness']:.2f} | "
               f"{metrics['Kurtosis']:.2f} | {metrics['25th Percentile']:.2f} | "
               f"{metrics['75th Percentile']:.2f} | {metrics['IQR']:.2f} | "
               f"{metrics['Range']:.2f} | {metrics['Min Angle']:.2f} | "
               f"{metrics['Max Angle']:.2f} |\n")
        rows += row

    # Create 'metrics' directory if it doesn't exist
    Path(f'{plot_folder}/metrics').mkdir(parents=True, exist_ok=True)
    
    # Write to markdown file
    with open(f'{plot_folder}/metrics/metrics.md', 'w') as f:
        f.write(header)
        f.write(separator)
        f.write(rows)

# Plot functions

def plot_histogram(angles, mlp_index):
    """
    Create a histogram plot of the angles and save it.

    Parameters:
    angles: List of angles to plot.
    mlp_index: Index of the MLP block for naming the file.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(angles, bins=30, color='blue', alpha=0.7)
    plt.title(f'Angle Distribution of MLP Block {mlp_index}')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Frequency')
    plt.grid(axis='y')

    # Calculate metrics
    mean = np.mean(angles)
    median = np.median(angles)
    mode = stats.mode(angles)[0]
    std_dev = np.std(angles)
    variance = np.var(angles)
    skewness = stats.skew(angles)
    kurtosis = stats.kurtosis(angles)
    percentiles = np.percentile(angles, [25, 50, 75])
    iqr = percentiles[2] - percentiles[0]
    data_range = np.max(angles) - np.min(angles)

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

    # Create the directory if it doesn't exist
    Path(plot_folder).mkdir(parents=True, exist_ok=True)
    plt.savefig(f'{plot_folder}/mlp_block_{mlp_index}_angles_histogram.png')
    plt.close()

    
def plot_histogram_logarithmic(angles, mlp_index):
    """
    Create a histogram plot of the angles and save it.

    Parameters:
    angles: List of angles to plot.
    mlp_index: Index of the MLP block for naming the file.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(angles, bins=30, color='blue', alpha=0.7)
    plt.title(f'Angle Distribution of MLP Block {mlp_index}')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Log(Frequency)')
    plt.grid(axis='y')

    # Set y-axis to logarithmic scale
    plt.yscale('log')

    # Calculate metrics
    mean = np.mean(angles)
    median = np.median(angles)
    mode = stats.mode(angles)[0]
    std_dev = np.std(angles)
    variance = np.var(angles)
    skewness = stats.skew(angles)
    kurtosis = stats.kurtosis(angles)
    percentiles = np.percentile(angles, [25, 50, 75])
    iqr = percentiles[2] - percentiles[0]
    data_range = np.max(angles) - np.min(angles)

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

    # Create the directory if it doesn't exist
    Path(plot_folder).mkdir(parents=True, exist_ok=True)
    plt.savefig(f'{plot_folder}/mlp_block_{mlp_index}_angles_histogram_logarithmic.png')
    plt.close()

def plot_boxplot(angles, mlp_index):
    """
    Create a horizontal box plot of the angles and save it.

    Parameters:
    angles: List of angles to plot.
    mlp_index: Index of the MLP block for naming the file.
    """
    plt.figure(figsize=(10, 6))
    
    # Create horizontal box plot
    plt.boxplot(angles, patch_artist=True, boxprops=dict(facecolor='skyblue', color='blue'),
                medianprops=dict(color='red'), whiskerprops=dict(color='blue'),
                capprops=dict(color='blue'), flierprops=dict(markerfacecolor='red', marker='o'),
                vert=False)  # Set vert=False for horizontal box plot

    plt.title(f'Box Plot of Angles for MLP Block {mlp_index}')
    plt.xlabel('Angle (degrees)')  # Change ylabel to xlabel for horizontal plot
    plt.grid(axis='x')  # Change grid to x-axis

    # Calculate metrics
    mean = np.mean(angles)
    median = np.median(angles)
    mode = stats.mode(angles)[0]  # Use [0][0] for a single mode
    std_dev = np.std(angles)
    variance = np.var(angles)
    skewness = stats.skew(angles)
    kurtosis = stats.kurtosis(angles)
    percentiles = np.percentile(angles, [25, 50, 75])
    iqr = percentiles[2] - percentiles[0]
    data_range = np.max(angles) - np.min(angles)

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

    plt.text(0.8, 0.7, metrics_text, transform=plt.gca().transAxes, fontsize=10,
             bbox=dict(facecolor='white', alpha=0.5), ha='center')  # Center align the text

    # Create the directory if it doesn't exist
    Path(plot_folder).mkdir(parents=True, exist_ok=True)
    plt.savefig(f'{plot_folder}/mlp_block_{mlp_index}_angles_boxplot.png')
    plt.close()



def create_histograms_for_mlp_blocks(mlp_blocks):
    """
    Loop through all MLP blocks, calculate angles, create histograms, and save them.

    Parameters:
    mlp_blocks: List of MLP blocks to process.
    """
    all_metrics = []
    for index, mlp_block in enumerate(mlp_blocks):
        angles = calculate_angles(mlp_block)
        plot_histogram(angles, index)
        plot_histogram_logarithmic(angles, index)
        plot_boxplot(angles, index)
        # Calculate metrics and save to markdown
        metrics = calculate_metrics(angles)
        all_metrics.append(metrics)

    save_metrics_to_markdown_table(all_metrics)

def create_hook(topk=100):
    def forward_hook(module, input, output):
        with torch.no_grad():
            # output shape: (batch, tokens, features)
            # Flatten activations per token, keep only top 100 per token
            activ = output.detach().clone()
            # activ shape: (batch, tokens, features)
            batch, tokens, features = activ.shape
            mask = torch.zeros_like(activ)
            values, indices = torch.topk(activ.abs(), topk, dim=2)
            mask.scatter_(2, indices, 1)
        return output * mask
    return forward_hook

# Parse the command line arguments
args = parser.parse_args()

# Load the pre-trained ViT model
model, weights = get_vit_imagenet()
# Update the save path for plots based on the model
plot_folder = args.model  # 'vitb16' or 'vitl16'


# for layer in model.encoder.layers:
#     #layer.mlp[0].register_forward_hook(create_hook(2500))
#     layer.mlp[3].register_forward_hook(create_hook(500))
#     pass

# Load and preprocess the input image
image = Image.open('cat_dog.jpg').convert('RGB')
# image_resized = image.resize([224,224])
# image_resized.convert('L').save('input_resized_grayscale.jpg')
# image_resized.save('input_resized.jpg')
input_tensor = weights.transforms()(image).unsqueeze(0).to("cuda")
# img_size = image.size  # (width, height)
summary(model, (input_tensor.shape))
# Assuming 'model' is your Vision Transformer model
mlp_blocks = []

# Access the encoder layers
for encoder_block in model.encoder.layers:
    # Each encoder block typically has an MLP layer named 'mlp'
    mlp_layer = encoder_block.mlp
    mlp_blocks.append(mlp_layer)

for index, mlp_block in enumerate(mlp_blocks):
        print(f"MLP Block {index}:")
        print(dir(mlp_block)) 
for name, param in mlp_block.named_parameters():
        print(f"Parameter Name: {name}, Shape: {param.shape}")

# Example usage (assuming 'mlp_blocks' is your list of MLP blocks):
create_histograms_for_mlp_blocks(mlp_blocks)

# Store the generated heatmaps
heatmaps = []

#############################################################
# Use topk activations for experimenting with epsilon rule:
#############################################################

input_tensor.grad = None  # Reset gradients
zennit_comp = LayerMapComposite([
    (torch.nn.Conv2d, z_rules.Epsilon()),
    (torch.nn.Linear, z_rules.Epsilon()),
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
heatmap = (input_tensor * input_tensor.grad).sum(1)

# Normalize relevance between [-1, 1] for plotting
heatmap = heatmap / abs(heatmap).max()
heatmap = heatmap.detach().cpu().numpy()

img = imgify(heatmap, vmin=-1, vmax=1)
img.convert('RGB').save('vit_heatmap.jpg')
