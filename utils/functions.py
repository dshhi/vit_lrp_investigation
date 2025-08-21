import numpy as np
from pathlib import Path
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vision_transformer
from skimage.metrics import structural_similarity

from zennit.image import imgify
from zennit.composites import LayerMapComposite
import zennit.rules as z_rules

import pdb
import PIL
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoImageProcessor, AutoModel
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from transformers import ViTImageProcessor, ViTModel
import timm
from urllib.request import urlopen

from typing import Tuple, Optional, Union, List
import copy

import requests

from tqdm import tqdm


# reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # might consider setting this to True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

## Load models

def get_model(args):
    """
    Load a model from selection.
    """

    if args.model in ('vitb16', 'vitl16'):
        if args.model == 'vitb16':
            weights = vision_transformer.ViT_B_16_Weights.IMAGENET1K_V1
            model = vision_transformer.vit_b_16(weights=weights)
        elif args.model == 'vitl16':
            weights = vision_transformer.ViT_L_16_Weights.IMAGENET1K_V1
            model = vision_transformer.vit_l_16(weights=weights)

        model.eval()
        model.to(args.device)

        # Deactivate gradients on parameters to save memory
        for param in model.parameters():
            param.requires_grad = False

        return model, weights

    # Load the tokenizer and model
    elif args.model in ('qwen3-8B','qwen3-4B','qwen3-0.6B', 'qwen2-0.5B','qwen2-7B'):
        if args.model == 'qwen2-0.5B':
            model_name = "Qwen/Qwen2-0.5B"
        if args.model == 'qwen2-7B':
            model_name = "Qwen/Qwen2-7B"
        elif args.model == 'qwen3-0.6B':
            model_name = "Qwen/Qwen3-0.6B"
        elif args.model == 'qwen3-4B':
            model_name = "Qwen/Qwen3-4B"
        elif args.model == 'qwen3-8B':
            model_name = "Qwen/Qwen3-8B"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map=args.device
        )

        model.eval()
        model.to(args.device)

        # Deactivate gradients on parameters to save memory
        for param in model.parameters():
            param.requires_grad = False

        return model, tokenizer

    elif args.model in ('vit-base-patch16-224','vit-large-patch16-224-in21k'):
        if (args.model == 'vit-large-patch16-224-in21k'):
            model_id = "google/vit-large-patch16-224-in21k"
        elif (args.model == 'vit-base-patch16-224'):
            model_id = "google/vit-base-patch16-224"

        model = ViTModel.from_pretrained(model_id, device_map=args.device).eval()

        # # Deactivate gradients on parameters to save memory
        # for param in model.parameters():
        #     param.requires_grad = False
        processor = ViTImageProcessor.from_pretrained(model_id)
            
        return model, processor

    elif args.model in ('dinov2-base','dinov2-large'):
        if (args.model == 'dinov2-base'):
            model_id = "facebook/dinov2-base"
        elif(args.model == 'dinov2-large'):
            model_id = "facebook/dinov2-large"
        model = AutoModel.from_pretrained(model_id, device_map=args.device).eval()
        processor = AutoImageProcessor.from_pretrained(model_id)

        return model, processor

    elif args.model in ('vit_giant_patch14_dinov2.lvd142m'):
        model_id = "timm/vit_giant_patch14_dinov2.lvd142m"
        model = timm.create_model(model_id, pretrained=True)
        return model

    elif args.model in ('gemma-3-12b-it', 'gemma-3-4b-it'):
        if (args.model == 'gemma-3-12b-it'):
            model_id = "google/gemma-3-12b-it"
        if (args.model == 'gemma-3-4b-it'):
            model_id = "google/gemma-3-4b-it"
        # pip install accelerate

        model = Gemma3ForConditionalGeneration.from_pretrained(
            # model_id, device_map="auto"
            model_id, device_map=args.device
        ).eval()

        processor = AutoProcessor.from_pretrained(model_id)

        return model, processor

def signed_margin_relu(W, X, alpha):
    D = W @ X
    r = W.norm(dim=1, keepdim=True)
    s = X.norm(dim=0, keepdim=True)
    margin = alpha * (r @ s)
    return D.sign() * F.relu(D.abs() - margin)

def create_signed_margin_relu_hook(alpha):
    def forward_hook(module, input, output):
        with torch.no_grad():
            # Get the input (first element if tuple)
            X = input[0] if isinstance(input, tuple) else input
            
            # Get weights from the module
            if hasattr(module, 'weight') and module.weight is not None:
                W = module.weight
                
                # Apply signed margin ReLU
                # Reshape input for matrix multiplication if needed
                original_shape = X.shape
                if X.dim() == 3:  # (batch, tokens, features)
                    batch, tokens, features = X.shape
                    X_flat = X.view(-1, features).T  # (features, batch*tokens)
                else:
                    X_flat = X.T if X.dim() == 2 else X
                
                # Apply signed_margin_relu
                result = signed_margin_relu(W, X_flat, alpha)
                
                # Reshape back to original format
                if X.dim() == 3:
                    result = result.T.view(batch, tokens, -1)
                elif X.dim() == 2:
                    result = result.T
                
                return result
            else:
                # If no weights found, return original output
                return output
                
    return forward_hook


class SignedMarginLinear(nn.Module):
    def __init__(self, in_features, out_features, alpha=0.1, bias=True, device=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.device = device
        
        # Initialize parameters on the correct device
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device))
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        original_shape = input.shape
        input = input.squeeze()
        D = input @ self.weight.t() # activations
        r = self.weight.norm(dim=1, keepdim=True) # radius / magnitude of weight vectors
        s = input.norm(dim=1, keepdim=True) # radius / magnitude of input vectors
        margin = self.alpha * (s * r.t())
        out = D.sign() * F.relu(D.abs() - margin)
        if self.bias is not None:
            out += self.bias.unsqueeze(0)  # Broadcast bias
        out = out.unsqueeze(0)
        return out


def replace_mlp_downstream_layer(args, mlp_block, alpha=0.1):
    """
    Replace the downstream linear layer in the MLP block with SignedMarginLinear layer.
    Only replaces the specific layer that corresponds to the one analyzed in calculate_angles_mlp_block.

    Parameters:
    args: Arguments containing model information
    mlp_block: The MLP block in which to replace the downstream linear layer
    alpha: Alpha parameter for SignedMarginLinear layer

    Returns:
    mlp_block: Modified MLP block with replaced downstream linear layer
    """
    
    # Import here to avoid circular imports if needed
    from utils.functions import SignedMarginLinear
    
    try:
        # Get device from args or infer from the mlp_block
        device = getattr(args, 'device', None)
        # Find and replace the downstream layer based on model type
        for name, module in mlp_block.named_modules():
            if args.model in ('vitb16', 'vitl16'):
                if name == '3':  # The layer at index 3
                    old_linear = module
                    new_linear = SignedMarginLinear(
                        in_features=old_linear.in_features,
                        out_features=old_linear.out_features,
                        alpha=alpha,
                        bias=old_linear.bias is not None,
                        device=device
                    )
                    with torch.no_grad():
                        new_linear.weight.copy_(old_linear.weight.to(device))
                        if old_linear.bias is not None:
                            new_linear.bias.copy_(old_linear.bias.to(device))
                    
                    mlp_block[3] = new_linear
                    # print(f"Replaced layer '3' for model {args.model} on device {device}")
                    break
                    
            elif args.model in ('qwen3-8B','qwen3-4B','qwen3-0.6B', 'qwen2-0.5B', 'qwen2-7B'):
                if name == 'down_proj':
                    old_linear = module
                    new_linear = SignedMarginLinear(
                        in_features=old_linear.in_features,
                        out_features=old_linear.out_features,
                        alpha=alpha,
                        bias=old_linear.bias is not None,
                        device=device
                    )
                    # Move to correct device and copy weights and bias
                    new_linear = new_linear.to(device)
                    with torch.no_grad():
                        new_linear.weight.copy_(old_linear.weight.to(device))
                        if old_linear.bias is not None:
                            new_linear.bias.copy_(old_linear.bias.to(device))
                    
                    # Replace the layer
                    mlp_block.down_proj = new_linear
                    print(f"Replaced 'down_proj' layer for model {args.model} on device {device}")
                    break
                    
            elif args.model in ('gemma-3-4b-it', 'gemma-3-12b-it'):
                if name == 'fc2':
                    old_linear = module
                    new_linear = SignedMarginLinear(
                        in_features=old_linear.in_features,
                        out_features=old_linear.out_features,
                        alpha=alpha,
                        bias=old_linear.bias is not None,
                        device=device
                    )
                    # Move to correct device and copy weights and bias
                    new_linear = new_linear.to(device)
                    with torch.no_grad():
                        new_linear.weight.copy_(old_linear.weight.to(device))
                        if old_linear.bias is not None:
                            new_linear.bias.copy_(old_linear.bias.to(device))
                    
                    # Replace the layer
                    mlp_block.fc2 = new_linear
                    print(f"Replaced 'fc2' layer for model {args.model} on device {device}")
                    break
                elif name == 'down_proj':
                    old_linear = module
                    new_linear = SignedMarginLinear(
                        in_features=old_linear.in_features,
                        out_features=old_linear.out_features,
                        alpha=alpha,
                        bias=old_linear.bias is not None,
                        device=device
                    )
                    # Move to correct device and copy weights and bias
                    new_linear = new_linear.to(device)
                    with torch.no_grad():
                        new_linear.weight.copy_(old_linear.weight.to(device))
                        if old_linear.bias is not None:
                            new_linear.bias.copy_(old_linear.bias.to(device))
                    
                    # Replace the layer
                    mlp_block.down_proj = new_linear
                    print(f"Replaced 'down_proj' layer for model {args.model} on device {device}")
                    break
                    
            elif args.model in ('vit-base-patch16-224', 'vit-large-patch16-224-in21k'):
                if name == 'dense':
                    old_linear = module
                    new_linear = SignedMarginLinear(
                        in_features=old_linear.in_features,
                        out_features=old_linear.out_features,
                        alpha=alpha,
                        bias=old_linear.bias is not None,
                        device=device
                    )
                    # Move to correct device and copy weights and bias
                    new_linear = new_linear.to(device)
                    with torch.no_grad():
                        new_linear.weight.copy_(old_linear.weight.to(device))
                        if old_linear.bias is not None:
                            new_linear.bias.copy_(old_linear.bias.to(device))
                    
                    # Replace the layer
                    mlp_block.dense = new_linear
                    print(f"Replaced 'dense' layer for model {args.model} on device {device}")
                    break
        
    except Exception as e:
        print(f"Error replacing downstream linear layer in MLP block: {e}")
        print(f"MLP block structure: {mlp_block}")
        
    return mlp_block

def replace_linear_layers_with_signed_margin(model, alpha=0.1, target_layers=None):
    """
    Replace specific Linear layers in the model with SignedMarginLinear layers.
    
    Args:
        model: The model to modify
        alpha: Alpha parameter for SignedMarginLinear
        target_layers: List of layer names/patterns to replace (e.g., ['mlp.3', 'output.dense'])
                      If None, replaces all Linear layers
    """
    def should_replace_layer(name, module):
        if not isinstance(module, nn.Linear):
            return False
        if target_layers is None:
            return True
        return any(target in name for target in target_layers)
    
    def replace_layers(module, name=""):
        for child_name, child_module in list(module.named_children()):
            full_name = f"{name}.{child_name}" if name else child_name
            
            if should_replace_layer(full_name, child_module):
                # Create new SignedMarginLinear layer
                new_layer = SignedMarginLinear(
                    in_features=child_module.in_features,
                    out_features=child_module.out_features,
                    alpha=alpha,
                    bias=child_module.bias is not None
                )
                
                # Copy weights and bias from original layer
                with torch.no_grad():
                    new_layer.weight.copy_(child_module.weight)
                    if child_module.bias is not None:
                        new_layer.bias.copy_(child_module.bias)
                
                # Replace the layer
                setattr(module, child_name, new_layer)
                print(f"Replaced {full_name} with SignedMarginLinear")
            else:
                # Recursively process child modules
                replace_layers(child_module, full_name)
    
    replace_layers(model)
    return model

def low_rank_svd_decomposition(weight_matrix: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform low-rank SVD decomposition on a weight matrix.
    
    Args:
        weight_matrix: Input weight matrix of shape (out_features, in_features)
        rank: Desired rank for low-rank approximation
        
    Returns:
        Tuple of (U_truncated, S_truncated, V_truncated)
    """
    # Perform SVD
    U, S, Vt = torch.linalg.svd(weight_matrix, full_matrices=False)
    
    # Truncate to desired rank
    rank = min(rank, min(U.shape[0], Vt.shape[0]))
    U_truncated = U[:, :rank]
    S_truncated = S[:rank]
    V_truncated = Vt[:rank, :]
    
    return U_truncated, S_truncated, V_truncated

def apply_low_rank_svd_to_mlp_block(mlp_block: nn.Module, rank: int, 
                                   inplace: bool = False) -> nn.Module:
    """
    Apply low-rank SVD to all Linear layers in an MLP block.
    
    Args:
        mlp_block: The MLP block containing Linear layers
        rank: Desired rank for low-rank approximation
        inplace: Whether to modify the original block or create a copy
        
    Returns:
        MLP block with low-rank approximated weights
    """
    if not inplace:
        mlp_block = copy.deepcopy(mlp_block)
    
    # Apply SVD to all Linear layers in the MLP block
    for name, layer in mlp_block.named_modules():
        if isinstance(layer, nn.Linear):
            with torch.no_grad():
                # Get original weight
                original_weight = layer.weight.data
                
                # Perform low-rank SVD
                U, S, V = low_rank_svd_decomposition(original_weight, rank)
                
                # Reconstruct low-rank approximation
                low_rank_weight = U @ torch.diag(S) @ V
                
                # Replace the weight
                layer.weight.data = low_rank_weight
                
                print(f"Applied rank-{rank} SVD to layer: {name}")
                print(f"Original shape: {original_weight.shape}, "
                      f"Compression ratio: {(U.shape[1] * (U.shape[0] + V.shape[1])) / original_weight.numel():.3f}")
    
    return mlp_block

def create_low_rank_mlp_blocks(mlp_blocks: List[nn.Module], rank: int, 
                              inplace: bool = False) -> List[nn.Module]:
    """
    Apply low-rank SVD to a list of MLP blocks.
    
    Args:
        mlp_blocks: List of MLP blocks
        rank: Desired rank for low-rank approximation
        inplace: Whether to modify original blocks or create copies
        
    Returns:
        List of MLP blocks with low-rank approximated weights
    """
    processed_blocks = []
    
    for i, mlp_block in enumerate(mlp_blocks):
        print(f"Processing MLP block {i+1}/{len(mlp_blocks)}")
        low_rank_block = apply_low_rank_svd_to_mlp_block(mlp_block, rank, inplace)
        processed_blocks.append(low_rank_block)
    
    return processed_blocks

def calculate_noise_metrics(ref_img_path, img_path):
    ref_image = cv2.imread(ref_img_path)
    gray_image = ref_image
    image = cv2.imread(img_path)
    
    # Calculate the noise
    noise = gray_image - image
    
    # Calculate the squared noise
    squared_noise = noise ** 2
    
    # Normalize the image for display purposes
    noise_normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    noise_image = Image.fromarray(noise_normalized.astype(np.uint8))
    
    # Calculate mean and standard deviation of the squared noise
    mean_squared_noise = np.mean(squared_noise)
    std_noise = np.std(noise)
    
    score = None
    return mean_squared_noise, std_noise, score

def create_reference_img(model,weights, input_tensor, output_folder):
    input_tensor.grad = None
    conv_gamma = 100
    lin_gamma = 1
    zennit_comp = LayerMapComposite([
        (torch.nn.Conv2d, z_rules.Gamma(conv_gamma)),
        (torch.nn.Linear, z_rules.Gamma(lin_gamma)),
    ])
    for layer in model.encoder.layers:
        #layer.mlp[0].register_forward_hook(create_hook(2500))
        layer.mlp[3].register_forward_hook(create_hook(100))
        pass
    # zennit_comp = LayerMapComposite([
    #     (torch.nn.Conv2d, z_rules.Epsilon()),
    #     (torch.nn.Linear, z_rules.Epsilon()),
    # ])
    zennit_comp.register(model)
    y = model(input_tensor.requires_grad_())
    _, top5_classes = torch.topk(y, 5, dim=1)
    top5_classes = top5_classes.squeeze(0).tolist()
    labels = weights.meta["categories"]
    top5_labels = [labels[class_idx] for class_idx in top5_classes]
    y[0, 156].backward()
    zennit_comp.remove()
    heatmap_gamma = (input_tensor * input_tensor.grad).sum(1)
    heatmap_gamma = heatmap_gamma / abs(heatmap_gamma).max()
    heatmap_gamma = heatmap_gamma.detach().cpu().numpy()
    img_gamma = imgify(heatmap_gamma, vmin=-1, vmax=1)
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True)
    img_gamma.convert('RGB').save(output_folder / 'vit_heatmap_reference.png')

def create_alpha_angle_img(args,input_tensor, output_folder):
    start, stop, step = args.alpha_range
    for alpha in tqdm(np.arange(start, stop, step)):
        model, weights = get_model(args)
        if args.model in ('vitb16', 'vitl16'):
            model, weights = get_model(args)
            mlp_blocks = []

            # Access the encoder layers
            for encoder_block in model.encoder.layers:
                # Each encoder block typically has an MLP layer named 'mlp'
                mlp_layer = encoder_block.mlp
                mlp_layer = replace_mlp_downstream_layer(args, mlp_layer, alpha)
                mlp_blocks.append(mlp_layer)

        elif model_name in ('vit-base-patch16-224','vit-large-patch16-224-in21k'):
            pass
        elif model_name in ('gemma-3-12b-it', 'gemma-3-4b-it'):
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

        input_tensor.grad = None
        # zennit_comp = LayerMapComposite([
        #     (torch.nn.Conv2d, z_rules.ZPlus()),
        #     (torch.nn.Linear, z_rules.Epsilon()),
        # ])
        zennit_comp = LayerMapComposite([
            (torch.nn.Conv2d, z_rules.ZPlus()),
            (torch.nn.Linear, z_rules.Epsilon()),
        ])
        zennit_comp.register(model)
        y = model(input_tensor.requires_grad_())
        _, top5_classes = torch.topk(y, 5, dim=1)
        top5_classes = top5_classes.squeeze(0).tolist()
        labels = weights.meta["categories"]
        top5_labels = [labels[class_idx] for class_idx in top5_classes]
        y[0, 156].backward()
        zennit_comp.remove()
        heatmap = (input_tensor * input_tensor.grad).sum(1)
        heatmap = heatmap / abs(heatmap).max()
        heatmap = heatmap.detach().cpu().numpy()
        img = imgify(heatmap, vmin=-1, vmax=1)
        output_folder = Path(output_folder)
        output_folder.mkdir(exist_ok=True)
        img.convert('RGB').save(output_folder / f'vit_heatmap_alpha{alpha:.4f}.png')

def calc_noise_array(args, output_folder):
    output_folder = Path(output_folder)
    ref_img_path = output_folder / "vit_heatmap_reference.png"
    results = []  # Store both alpha and metrics together
    start, stop, step = args.alpha_range
    
    for alpha in tqdm(np.arange(start, stop, step)):
        img_path = output_folder / f"vit_heatmap_alpha{alpha:.4f}.png"
        try:
            mean_noise, std_noise, score = calculate_noise_metrics(ref_img_path, img_path)
            # Store alpha and metrics together to ensure they stay synchronized
            results.append({
                'alpha': alpha,
                'mean_noise': mean_noise,
                'std_noise': std_noise,
                'score': score
            })
        except Exception as e:
            print(f"Skipping alpha {alpha:.4f}: {str(e)}")
            continue
    
    return results


def plot_noise_metrics(results, metric, model, output_folder):
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True)
    
    if not results:
        print("No data to plot!")
        return
    
    # Extract alpha values and corresponding metrics
    alpha_values = [entry['alpha'] for entry in results]
    noise_metrics = [entry[metric] for entry in results]
    
    print(f"Plotting {len(alpha_values)} points")
    print(f"Alpha range: {min(alpha_values):.4f} to {max(alpha_values):.4f}")
    
    plt.style.use('ggplot')
    plt.figure(figsize=(12, 8), dpi=300)
    
    # Plot using alpha values as x-axis
    plt.plot(alpha_values, noise_metrics, linestyle='-', color='#FF5733', linewidth=1.5, 
             label=metric.replace('_', ' ').title())
    
    # Find minima and their corresponding alpha values
    minima_indices = np.argsort(noise_metrics)[:3]
    minima_values = [noise_metrics[i] for i in minima_indices]
    minima_alphas = [alpha_values[i] for i in minima_indices]
    
    # Plot minima points
    plt.scatter(minima_alphas, minima_values, color='blue', zorder=5, label='Top 3 Minima')
    
    # Annotate minima points with alpha values and metric values
    for i, (alpha, value) in enumerate(zip(minima_alphas, minima_values)):
        plt.text(alpha, value, f'α: {alpha:.4f}\nVal: {value:.2f}', 
                 fontsize=10, ha='center', va='bottom', color='blue',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.title(f'{metric.replace("_", " ").title()} vs Alpha Values', fontsize=16)
    plt.xlabel('Alpha Value', fontsize=14)
    plt.ylabel(metric.replace('_', ' ').title(), fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)
    
    img_path = output_folder / f'noise_plot_{model}_{metric}.png'
    plt.legend()
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    plt.show()  # Remove this if you don't want to display the plot

def create_topk_img(input_tensor, output_folder):
    for k in tqdm(range(768)):
        model, weights = get_vit_imagenet()
        for layer in model.encoder.layers:
            layer.mlp[3].register_forward_hook(create_hook(k))
        input_tensor.grad = None
        zennit_comp = LayerMapComposite([
            (torch.nn.Conv2d, z_rules.ZPlus()),
            (torch.nn.Linear, z_rules.Epsilon()),
        ])
        zennit_comp.register(model)
        y = model(input_tensor.requires_grad_())
        _, top5_classes = torch.topk(y, 5, dim=1)
        top5_classes = top5_classes.squeeze(0).tolist()
        labels = weights.meta["categories"]
        top5_labels = [labels[class_idx] for class_idx in top5_classes]
        y[0, 156].backward()
        zennit_comp.remove()
        heatmap = (input_tensor * input_tensor.grad).sum(1)
        heatmap = heatmap / abs(heatmap).max()
        heatmap = heatmap.detach().cpu().numpy()
        img = imgify(heatmap, vmin=-1, vmax=1)
        output_folder.mkdir(exist_ok=True)
        img.convert('RGB').save(output_folder / f'vit_heatmap_top{k}.png')

def calculate_angles_mlp_block(args,mlp_block):
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
            if args.model in ('vitb16', 'vitl16'):
                if name == '3.weight':
                    weights2 = param.data.cpu().numpy()
                    break
            elif args.model in ('qwen3-8B','qwen3-4B','qwen3-0.6B','qwen2-0.5B','qwen2-7B'):
                if name == 'down_proj.weight':
                    weights2 = param.data.to(torch.float32)
                    weights2 = weights2.data.cpu().numpy()           
                    break
            elif args.model in ('dinov2-base','dinov2-large'):
                if name == 'fc2.weight':
                    weights2 = param.data.to(torch.float32)
                    weights2 = weights2.data.cpu().numpy()           
                    break
            elif args.model in ('vit_giant_patch14_dinov2.lvd142m'):
                if name == 'fc2.weight':
                    weights2 = param.data.to(torch.float32)
                    weights2 = weights2.data.cpu().numpy()           
                    break
            elif args.model in ('gemma-3-4b-it','gemma-3-12b-it'):
                if name == 'fc2.weight':
                    weights2 = param.data.cpu().numpy()
                    break
                if name == 'down_proj.weight':
                    weights2 = param.data.cpu().numpy()
                    break
            elif args.model in ('vit-base-patch16-224','vit-large-patch16-224-in21k'):
                if name == 'dense.weight':
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


def save_metrics_to_markdown_table(args, all_metrics):
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
    Path(f'{args.results_dir}/metrics').mkdir(parents=True, exist_ok=True)
    
    # Write to markdown file
    with open(f'{args.results_dir}/metrics/metrics.md', 'w') as f:
        f.write(header)
        f.write(separator)
        f.write(rows)

# Plot functions

def plot_histogram(args, angles, mlp_index):
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
    Path(args.plot_path).mkdir(parents=True, exist_ok=True)
    plt.savefig(f'{args.plot_path}/mlp_block_{mlp_index}_angles_histogram.png')
    plt.close()

    
def plot_histogram_logarithmic(args, angles, mlp_index):
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
    Path(args.plot_path).mkdir(parents=True, exist_ok=True)
    plt.savefig(f'{args.plot_path}/mlp_block_{mlp_index}_angles_histogram_logarithmic.png')
    plt.close()

def plot_boxplot(args, angles, mlp_index):
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
    Path(args.plot_path).mkdir(parents=True, exist_ok=True)
    plt.savefig(f'{args.plot_path}/mlp_block_{mlp_index}_angles_boxplot.png')
    plt.close()



def create_histograms_for_mlp_blocks(args, mlp_blocks):
    """
    Loop through all MLP blocks, calculate angles, create histograms, and save them.

    Parameters:
    mlp_blocks: List of MLP blocks to process.
    """
    all_metrics = []

    for index, mlp_block in tqdm(enumerate(mlp_blocks),
                                    total=len(mlp_blocks),
                                    desc="Processing MLP Blocks"):
        angles = calculate_angles_mlp_block(args, mlp_block)
        plot_histogram(args, angles, index)
        plot_histogram_logarithmic(args, angles, index)
        plot_boxplot(args, angles, index)
        # Calculate metrics and save to markdown
        metrics = calculate_metrics(angles)
        all_metrics.append(metrics)

    save_metrics_to_markdown_table(args, all_metrics)

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
