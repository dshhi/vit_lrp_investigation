from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

import numpy as np
import torch
import itertools
import sys
import argparse
from pathlib import Path
import pdb

from utils.logging import get_logger
# model loading
from utils.functions import get_model 
from utils.functions import set_seed
from utils.functions import create_histograms_for_mlp_blocks
# hooks
from utils.functions import create_hook

import numpy as np
import torch
import itertools
import sys
import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))

from utils.logging import get_logger
from utils.args import get_public_config
from utils.functions import set_seed

from torchvision.models import vision_transformer
from zennit.image import imgify
from zennit.composites import LayerMapComposite
import zennit.rules as z_rules
from tqdm import tqdm
from lxt.efficient import monkey_patch, monkey_patch_zennit
from PIL import Image
from torchinfo import summary

monkey_patch(vision_transformer, verbose=True)
monkey_patch_zennit(verbose=True)

def get_config():
    parser = get_public_config()
    # Logger
    # log_dir = "{}/{}/{}/seq_len_{}_pred_len_{}_bs_{}/".format(
    #     base_dir,
    #     args.model_name + "_without_sam",
    #     args.dataset,
    #     args.seq_len,
    #     args.horizon,
    #     args.batch_size,
    # )
    args = parser.parse_args()
    results_dir = SCRIPT_DIR / "results" / f"{args.model}"
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

    if model_name in ('vitb16', 'vitl16'):
        model, weights = get_vit_imagenet(args)
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

        # Example usage (assuming 'mlp_blocks' is your list of MLP blocks):
        create_histograms_for_mlp_blocks(args, mlp_blocks)

        # # Store the generated heatmaps
        # heatmaps = []
        #
        # #############################################################
        # # Use topk activations for experimenting with epsilon rule:
        # #############################################################
        #
        # input_tensor.grad = None  # Reset gradients
        # zennit_comp = LayerMapComposite([
        #     (torch.nn.Conv2d, z_rules.Epsilon()),
        #     (torch.nn.Linear, z_rules.Epsilon()),
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
        # heatmap = (input_tensor * input_tensor.grad).sum(1)
        #
        # # Normalize relevance between [-1, 1] for plotting
        # heatmap = heatmap / abs(heatmap).max()
        # heatmap = heatmap.detach().cpu().numpy()
        #
        # img = imgify(heatmap, vmin=-1, vmax=1)
        # img.convert('RGB').save('vit_heatmap.jpg')
    elif model_name in ('Qwen3-0.6B'):
        # Prepare input text
        model,tokenizer = get_model(args)
        # prompt = "Polar bears live in"
        #
        # messages = [
        #     {"role": "user", "content": prompt}
        # ]
        # text = tokenizer.apply_chat_template(
        #     messages,
        #     tokenize=False,
        #     add_generation_prompt=True,
        #     enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
        # )
        #
        # model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        #
        #
        # # Use torchinfo to get the summary
        # # summary(model, (model_inputs.shape))
        # # summary(model, input_size=(1, inputs_ids['input_ids'].shape[1]))
        #
        # # Forward pass
        # #
        # generated_ids = model.generate(
        #                     **model_inputs,
        #                     max_new_tokens=32768
        #                 )
        # output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        # output = model(input_ids)

        # summary(model, input_size=(1, input_ids['input_ids'].shape[1]))
        # summary(model, (input_ids.shape))

        mlp_blocks = []
        # pdb.set_trace()

        # Access the encoder layers
        for model_layer in model.model.layers:
            # Each encoder block typically has an MLP layer named 'mlp'
            mlp_layer = model_layer.mlp
            mlp_blocks.append(mlp_layer)

        # Example usage (assuming 'mlp_blocks' is your list of MLP blocks):
        create_histograms_for_mlp_blocks(args, mlp_blocks)

        # # Get the top 5 predictions (if applicable)
        # logits = output.logits
        # _, top5_classes = torch.topk(logits, 5, dim=-1)
        # top5_classes = top5_classes.squeeze(0).tolist()
        #
        # # Assuming you have a way to map class indices to labels
        # # Here, we would need a specific label mapping depending on your task
        # # top5_labels = [labels[class_idx] for class_idx in top5_classes]
        # # labels = ["label1", "label2", ...]  # Define your labels
        #
        # # Print the top 5 predictions
        # for i, class_idx in enumerate(top5_classes):
        #     print(f'Top {i+1} predicted class index: {class_idx}')  # Print class indices
        #
        # # Backward pass for the highest probability class (if needed)
        # # If you want to compute gradients, you need to ensure that requires_grad=True
        # logits = logits.squeeze(0)
        # logits.requires_grad_()
        # logits[0, top5_classes[0][0]].backward()
    if model_name in ('gemma-3-12b-it', 'gemma-3-4b-it'):
        model, processor = get_model(args)
        args.results_dir = args.results_dir / "vision_model"
        args.plot_path = args.results_dir / "plots"
        vision_model_mlp_blocks = []

        # Access the encoder layers of the vision_tower
        for model_layer in model.model.vision_tower.vision_model.encoder.layers:
            # Each encoder block typically has an MLP layer named 'mlp'
            mlp_layer = model_layer.mlp
            mlp_blocks.append(mlp_layer)

        # Example usage (assuming 'mlp_blocks' is your list of MLP blocks):
        create_histograms_for_mlp_blocks(args, mlp_blocks)
        pdb.set_trace()

        args.results_dir = args.results_dir / "language_model"
        args.plot_path = args.results_dir / "plots"
        language_model_mlp_blocks = []

        # Access the encoder layers of the vision_tower
        for model_layer in model.model.language_model.layers:
            # Each encoder block typically has an MLP layer named 'mlp'
            mlp_layer = model_layer.mlp
            mlp_blocks.append(mlp_layer)

        # Example usage (assuming 'mlp_blocks' is your list of MLP blocks):
        create_histograms_for_mlp_blocks(args, mlp_blocks)


if __name__ == "__main__":
    main()
