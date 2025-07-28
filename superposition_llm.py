from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

import numpy as np
import torch
import itertools
import sys
import argparse
from pathlib import Path
import pdb
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))

from utils.logging import get_logger
# model loading
from utils.functions import get_llm_model
from utils.functions import set_seed
from utils.functions import create_histograms_for_mlp_blocks_qwen
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
from utils.functions import get_vit_imagenet

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
    results_dir = SCRIPT_DIR / "results" / f"{args.llm_model}"
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
    model_name = args.llm_model
    results_dir = f"{model_name}_results"
    plots_dir = f"{results_dir}_results"

    # Create directories if they don't exist
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    Path(plots_dir).mkdir(parents=True, exist_ok=True)

    # Prepare input text
    tokenizer ,model = get_llm_model(args)
    prompt = "Polar bears live in"

    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)


    # Use torchinfo to get the summary
    # summary(model, (model_inputs.shape))
    # summary(model, input_size=(1, inputs_ids['input_ids'].shape[1]))

    # Forward pass
    #
    generated_ids = model.generate(
                        **model_inputs,
                        max_new_tokens=32768
                    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

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
    create_histograms_for_mlp_blocks_qwen(args, mlp_blocks)

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


if __name__ == "__main__":
    main()
