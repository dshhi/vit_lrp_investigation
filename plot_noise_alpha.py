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
from utils.functions import create_low_rank_mlp_blocks
# hooks
from utils.functions import create_reference_img,create_alpha_angle_img, calc_noise_array,plot_noise_metrics

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
    parser.add_argument('--create_ref_img', action='store_true', help='Create reference image')
    parser.add_argument('--create_alpha_img', action='store_true', help='Create top-k images')
    parser.add_argument('--noise_metric', type=str, choices=['mean_noise', 'std_noise', 'score'], default='mean_noise', help='Noise metric to plot')
    parser.add_argument('--alpha_range', nargs=3, type=float, metavar=('START', 'STOP', 'STEP'), 
                       default=[0.0001, 0.1, 0.0001],
                       help='Range parameters: start stop step (default: 0.001 0.1 0.001)')
    
    args = parser.parse_args()
    results_dir = SCRIPT_DIR / "results"
    #language models
    if args.model in ('qwen3-0.6B', 'qwen2-0.5B','qwen2-7B',
                      ):
        results_dir = results_dir / "language" / f"{args.model}"
    #vision models
    elif args.model in ('vitb16', 'vitl16','vit-base-patch16-224','vit-large-patch16-224-in21k',
                        ):
        results_dir = results_dir / "vision" / f"{args.model}"
    #multi-modal models
    elif args.model in ('gemma-3-12b-it', 'gemma-3-4b-it',
                        ):
        # save path gets handled in function for vision and language part
        # separately
        pass

    results_dir = Path(str(results_dir) + "/noise")

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
    image = Image.open('cat_dog.jpg').convert('RGB')
    model, weights = get_model(args)
    input_tensor = weights.transforms()(image).unsqueeze(0).to(args.device)

    # if args.create_ref_img:
    #     create_reference_img(model,weights,input_tensor, args.plot_path)
    # if args.create_alpha_img:
    #     create_alpha_angle_img(args,input_tensor, args.plot_path)

    results = calc_noise_array(args, args.plot_path)
    plot_noise_metrics(results, args.noise_metric, args.model, args.plot_path)

if __name__ == "__main__":
    main()
