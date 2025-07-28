from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

import pdb


# Setup argument parsing
parser = argparse.ArgumentParser(description='Select ViT model and output directory.')
parser.add_argument('--model', choices=['Qwen3-0.6B'], required=True, help='Select the model to use.')
args = parser.parse_args()

# Set up results directory based on model
model_name = args.model
results_dir = f"{model_name}_results"
plots_dir = f"{results_dir}/plots"

# Create directories if they don't exist
Path(results_dir).mkdir(parents=True, exist_ok=True)
Path(plots_dir).mkdir(parents=True, exist_ok=True)



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

    # Create 'plots' directory if it doesn't exist
    Path('plots').mkdir(parents=True, exist_ok=True)
    plt.savefig(f'plots/mlp_block_{mlp_index}_angles_histogram.png')
    plt.close()



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

    # Create 'plots' directory if it doesn't exist
    Path('plots').mkdir(parents=True, exist_ok=True)
    plt.savefig(f'plots/mlp_block_{mlp_index}_angles_histogram.png')
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

    # Create 'plots' directory if it doesn't exist
    Path('plots').mkdir(parents=True, exist_ok=True)
    plt.savefig(f'plots/mlp_block_{mlp_index}_angles_histogram_logarithmic.png')
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

    # Create 'plots' directory if it doesn't exist
    Path('plots').mkdir(parents=True, exist_ok=True)
    plt.savefig(f'plots/mlp_block_{mlp_index}_angles_boxplot.png')
    plt.close()


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
    Path('metrics').mkdir(parents=True, exist_ok=True)
    
    # Write to markdown file
    with open('metrics/metrics.md', 'w') as f:
        f.write(header)
        f.write(separator)
        f.write(rows)

def create_histograms_for_mlp_blocks(mlp_blocks):
    """
    Loop through all MLP blocks, calculate angles, create histograms, and save them.

    Parameters:
    mlp_blocks: List of MLP blocks to process.
    """
    all_metrics = []
    for index, mlp_block in enumerate(mlp_blocks):
        angles = calculate_angles(mlp_block)
        pdb.set_trace()
        # plot_histogram(angles, index)
        # plot_histogram_logarithmic(angles, index)
        # plot_boxplot(angles, index)
        # Calculate metrics and save to markdown
        metrics = calculate_metrics(angles)
        all_metrics.append(metrics)

    save_metrics_to_markdown_table(all_metrics)


# load the tokenizer and the model

def get_model(device="cuda"):
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )

    model.eval()
    model.to(device)

    # Deactivate gradients on parameters to save memory
    for param in model.parameters():
        param.requires_grad = False

    return tokenizer,model


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

# Register hooks for Llama-2 model (adjust based on layer structure)
# for layer in model.transformer.h:  # Assuming Llama-2 has a similar layer structure
#     layer.register_forward_hook(create_hook(100))

# Prepare input text
tokenizer,model=get_model(device="cuda")
input_text = "Polar bears live in"
input_ids = tokenizer(input_text, return_tensors='pt').input_ids.to("cuda")

# Forward pass
output = model(input_ids)

# Get the top 5 predictions (if applicable)
logits = output.logits
_, top5_classes = torch.topk(logits, 5, dim=-1)
top5_classes = top5_classes.squeeze(0).tolist()

# Assuming you have a way to map class indices to labels
# Here, we would need a specific label mapping depending on your task
# labels = ["label1", "label2", ...]  # Define your labels
# top5_labels = [labels[class_idx] for class_idx in top5_classes]

# Print the top 5 predictions
for i, class_idx in enumerate(top5_classes):
    print(f'Top {i+1} predicted class index: {class_idx}')  # Print class indices

# Backward pass for the highest probability class (if needed)
# If you want to compute gradients, you need to ensure that requires_grad=True
logits = logits.squeeze(0)
logits.requires_grad_()
logits[0, top5_classes[0][0]].backward()

# Note: Llama-2 doesn't have a concept of heatmaps like CNNs, so that part is omitted.
