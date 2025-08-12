import torch
import itertools
from PIL import Image
from torchvision.models import vision_transformer

from zennit.image import imgify
from zennit.composites import LayerMapComposite
import zennit.rules as z_rules

from lxt.efficient import monkey_patch, monkey_patch_zennit
import PIL
import torchvision.transforms as transforms
from utils.args import get_public_config
from utils.functions import get_model 

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))

from utils.logging import get_logger
from utils.args import get_public_config
from utils.functions import set_seed
from utils.functions import create_signed_margin_relu_hook 


import pdb

monkey_patch(vision_transformer, verbose=True)
monkey_patch_zennit(verbose=True)

def get_config():
    parser = get_public_config()
    args = parser.parse_args()
    results_dir = SCRIPT_DIR / "results" / "inference_angular_threshold"
    #vision models
    if args.model in ('vitb16', 'vitl16','vit-base-patch16-224','vit-large-patch16-224-in21k',
                        ):
        results_dir = results_dir / "vision" / f"{args.model}"
    #multi-modal models
    elif args.model in ('gemma-3-12b-it', 'gemma-3-4b-it',
                        ):
        # save path gets handled in function for vision and language part
        # separately
        pass

    plot_dir = f"{results_dir}/plots"
    args.results_dir = results_dir
    args.plot_path = plot_dir
    log_dir = results_dir

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger(log_dir, __name__, "record_s{}.log".format(args.seed))
    logger.info(args)

    return args, log_dir, logger, plot_dir

def main():
    args, log_dir, logger, plot_dir = get_config()
    set_seed(args.seed)

    # Set up results directory based on model
    model_name = args.model
    
    # Hook parameters
    alpha = args.alpha if hasattr(args, 'alpha') else 0.1  # Default alpha value

    if model_name in ('vitb16', 'vitl16'):
        model, weights = get_model(args)
        mlp_blocks = []

        # Access the encoder layers
        for encoder_block in model.encoder.layers:
            # Each encoder block typically has an MLP layer named 'mlp'
            mlp_layer = encoder_block.mlp
            mlp_blocks.append(mlp_layer)
            
            # Register hook on the MLP layer (adjust index as needed)
            if hasattr(mlp_layer, '__getitem__') and len(mlp_layer) > 3:
                pdb.set_trace()
                mlp_layer[3].register_forward_hook(create_signed_margin_relu_hook(alpha))
            else:
                # If mlp_layer is not indexable, register on the whole layer
                mlp_layer.register_forward_hook(create_signed_margin_relu_hook(alpha))

    elif model_name in ('vit-base-patch16-224','vit-large-patch16-224-in21k'):
        model, processor = get_model(args)

        mlp_blocks = []

        # Access the encoder layers
        for model_layer in model.encoder.layer:
            mlp_layer = model_layer.output
            mlp_blocks.append(mlp_layer)
            
            # Register hook on the output layer
            mlp_layer.register_forward_hook(create_signed_margin_relu_hook(alpha))

    elif model_name in ('gemma-3-12b-it', 'gemma-3-4b-it'):
        model, processor = get_model(args)
        results_dir = args.results_dir
        args.results_dir = results_dir / "vision" / f"{args.model}_vision"
        args.plot_path = args.results_dir / "plots"
        vision_model_mlp_blocks = []

        # Access the encoder layers of the vision_tower
        for model_layer in model.model.vision_tower.vision_model.encoder.layers:
            # Each encoder block typically has an MLP layer named 'mlp'
            mlp_layer = model_layer.mlp
            vision_model_mlp_blocks.append(mlp_layer)
            
            # Register hook on the MLP layer (adjust index as needed)
            if hasattr(mlp_layer, '__getitem__') and len(mlp_layer) > 3:
                mlp_layer[3].register_forward_hook(create_signed_margin_relu_hook(alpha))
            else:
                # If mlp_layer is not indexable, register on the whole layer
                mlp_layer.register_forward_hook(create_signed_margin_relu_hook(alpha))

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
    # zennit_comp = LayerMapComposite([
    #     (torch.nn.Conv2d, z_rules.ZPlus()),
    #     (torch.nn.Linear, z_rules.Epsilon()),
    # ])
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

    args.results_dir.mkdir(parents=True, exist_ok=True)
    img = imgify(heatmap, vmin=-1, vmax=1)
    save_dir = args.results_dir / "plots"
    save_dir.mkdir(parents=True, exist_ok=True)
    img.convert('RGB').save(f'{save_dir}/vit_heatmap_alpha{args.alpha}.jpg')
    img_resized = img.resize(img_size, PIL.Image.Resampling.LANCZOS).convert('RGB')

    # Save the resized image with a better quality
    img_resized.save(f'{save_dir}/vit_heatmap_alpha{args.alpha}_orig_size.jpg', quality=100)  #

    #
    #
    # #############################################################
    # # Experiment with different gamma values for Conv2d and Linear layers
    # # Gamma is a hyperparameter in LRP that controls how much positive vs. negative
    # # contributions are considered in the explanation
    # #############################################################
    # input_tensor.grad = None  # Reset gradients
    #
    # # Define rules for the Conv2d and Linear layers using 'zennit'
    # # LayerMapComposite maps specific layer types to specific LRP rule implementations
    # conv_gamma = 100
    # lin_gamma = 1
    # print("Gamma Conv2d:", conv_gamma, "Gamma Linear:", lin_gamma)
    # zennit_comp = LayerMapComposite([
    #     (torch.nn.Conv2d, z_rules.Gamma(conv_gamma)),
    #     (torch.nn.Linear, z_rules.Gamma(lin_gamma)),
    # ])
    #
    # # Register the composite rules with the model
    # zennit_comp.register(model)
    #
    # # Forward pass with gradient tracking enabled
    # y = model(input_tensor.requires_grad_())
    #
    # # Get the top 5 predictions
    # _, top5_classes = torch.topk(y, 5, dim=1)
    # top5_classes = top5_classes.squeeze(0).tolist()
    #
    # # Get the class labels
    # labels = weights.meta["categories"]
    # top5_labels = [labels[class_idx] for class_idx in top5_classes]
    #
    # # Print the top 5 predictions and their labels
    # for i, class_idx in enumerate(top5_classes):
    #     print(f'Top {i+1} predicted class: {class_idx}, label: {top5_labels[i]}')
    #
    # # Backward pass for the highest probability class
    # # This initiates the LRP computation through the network
    # y[0, 156].backward()
    #
    # # Remove the registered composite to prevent interference in future iterations
    # zennit_comp.remove()
    #
    # # Calculate the relevance by computing Gradient * Input
    # # This is the final step of LRP to get the pixel-wise explanation
    # heatmap_gamma = (input_tensor * input_tensor.grad).sum(1)
    #
    # # Normalize relevance between [-1, 1] for plotting
    # heatmap_gamma = heatmap_gamma / abs(heatmap_gamma).max()
    #
    # heatmap_gamma = heatmap_gamma.detach().cpu().numpy()
    #
    # # Visualize all heatmaps in a grid (3×5) and save to a file
    # # vmin and vmax control the color mapping range
    # #imgify(heatmaps, vmin=-1, vmax=1).save('vit_heatmap.png')
    # img_gamma = imgify(heatmap_gamma, vmin=-1, vmax=1)
    # img_gamma.convert('RGB').save('vit_heatmap_gamma.jpg')
    # #imgify(heatmaps, vmin=-1, vmax=1).save('vit_heatmap.png')
    # img_gamma_resized = img_gamma.resize(img_size, PIL.Image.Resampling.LANCZOS).convert('RGB')
    #
    #
    #
    # # Save the resized image with a better quality
    # img_gamma_resized.save('vit_heatmap_gamma_orig_size.jpg', quality=100)  #
    # #imgify(heatmaps[0], vmin=-1, vmax=1).save('vit_heatmap.png')
    #
if __name__ == "__main__":
    main()
