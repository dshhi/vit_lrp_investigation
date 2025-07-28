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
import cv2
import numpy as np
from skimage.metrics import structural_similarity
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

monkey_patch(vision_transformer, verbose=True)
monkey_patch_zennit(verbose=True)

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

def get_vit_imagenet(device="cuda", model ="vitb16"):
    if model == "vitb16":
        weights = vision_transformer.ViT_B_16_Weights.IMAGENET1K_V1
        model = vision_transformer.vit_b_16(weights=weights)
    elif model == "vitl16":
        weights = vision_transformer.ViT_L_16_Weights.IMAGENET1K_V1
        model = vision_transformer.vit_l_16(weights=weights)
    model.eval()
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    return model, weights

def create_hook(topk=100):
    def forward_hook(module, input, output):
        with torch.no_grad():
            activ = output.detach().clone()
            batch, tokens, features = activ.shape
            values, indices = torch.topk(activ.abs(), topk, dim=2)
            mask = torch.zeros_like(activ)
            mask.scatter_(2, indices, 1)
        return output * mask
    return forward_hook

# def create_hook(topk=100):
#     def forward_hook(module, input, output):
#         # output shape: (batch, tokens, features)
#         with torch.no_grad():
#             values, indices = torch.topk(output.abs(), topk, dim=2)
#             mask = torch.zeros_like(output)
#             mask.scatter_(2, indices, 1)
#
#         # Keep gradients for top-k, detach the rest
#         detached_output = output.detach()
#         output = torch.where(mask.bool(), output, detached_output)
#         return output
#     return forward_hook

def create_reference_img(input_tensor, output_folder):
    input_tensor.grad = None
    conv_gamma = 100
    lin_gamma = 1
    # zennit_comp = LayerMapComposite([
    #     (torch.nn.Conv2d, z_rules.Gamma(conv_gamma)),
    #     (torch.nn.Linear, z_rules.Gamma(lin_gamma)),
    # ])
    for layer in model.encoder.layers:
        #layer.mlp[0].register_forward_hook(create_hook(2500))
        layer.mlp[3].register_forward_hook(create_hook(100))
        pass
    zennit_comp = LayerMapComposite([
        (torch.nn.Conv2d, z_rules.Epsilon()),
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
    heatmap_gamma = (input_tensor * input_tensor.grad).sum(1)
    heatmap_gamma = heatmap_gamma / abs(heatmap_gamma).max()
    heatmap_gamma = heatmap_gamma.detach().cpu().numpy()
    img_gamma = imgify(heatmap_gamma, vmin=-1, vmax=1)
    output_folder.mkdir(exist_ok=True)
    img_gamma.convert('RGB').save(output_folder / 'vit_heatmap_reference.png')

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

def calc_noise_array(output_folder):
    ref_img_path = output_folder / "vit_heatmap_reference.png"
    noise_array = []
    for k in tqdm(range(768)):
        img_path = output_folder / f"vit_heatmap_top{k}.png"
        mean_noise, std_noise, score = calculate_noise_metrics(ref_img_path, img_path)
        noise_array.append([mean_noise, std_noise, score])
    return noise_array

def plot_noise_metrics(noise, metric, model):
    if metric == 'mean_noise':
        noise_metrics = [entry[0] for entry in noise]
    elif metric == 'std_noise':
        noise_metrics = [entry[1] for entry in noise]
    elif metric == 'score':
        noise_metrics = [entry[2] for entry in noise]

    plt.style.use('ggplot')
    plt.figure(figsize=(12, 8), dpi=300)
    plt.plot(noise_metrics, linestyle='-', color='#FF5733', linewidth=1.5, label=metric.replace('_', ' ').title())
    minima_indices = np.argsort(noise_metrics)[:3]
    minima_values = [noise_metrics[i] for i in minima_indices]
    plt.scatter(minima_indices, minima_values, color='blue', zorder=5, label='Top 3 Minima')
    for i, idx in enumerate(minima_indices):
        plt.text(idx, minima_values[i], f'Idx: {idx}, Val: {minima_values[i]:.2f}', 
                 fontsize=12, ha='center', va='bottom', color='blue')
    plt.title('Noise Metrics', fontsize=16)
    plt.xlabel('Index', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)
    plot_filename = f'noise_plot_{model}.png'
    plt.legend()
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')

def main():
    parser = argparse.ArgumentParser(description='Process some images with ViT.')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cuda', help='Device to use')
    parser.add_argument('--model', type=str, choices=['vitb16', 'vitl16'], default='vitb16', help='Model to use')
    parser.add_argument('--create_ref_img', action='store_true', help='Create reference image')
    parser.add_argument('--create_topk_img', action='store_true', help='Create top-k images')
    parser.add_argument('--noise_metric', type=str, choices=['mean_noise', 'std_noise', 'score'], default='mean_noise', help='Noise metric to plot')

    args = parser.parse_args()

    output_folder = Path(f'{args.model}_noise_img')
    
    global model, weights
    model, weights = get_vit_imagenet(args.device, args.model)
    
    image = Image.open('cat_dog.jpg').convert('RGB')
    input_tensor = weights.transforms()(image).unsqueeze(0).to(args.device)

    if args.create_ref_img:
        create_reference_img(input_tensor, output_folder)
    # if args.create_topk_img:
    #     create_topk_img(input_tensor, output_folder)
    #
    # noise_array = calc_noise_array(output_folder)
    # plot_noise_metrics(noise_array, args.noise_metric, args.model)

if __name__ == '__main__':
    main()
