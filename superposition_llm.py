from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

import pdb
SCRIPT_DIR = Path(__file__).resolve().parent

# model loading
from utils import get_llm_model
# computation utils
from utils import calculate_angles, calculate_metrics, save_metrics_to_markdown_table
# plotting utils
from utils import plot_histogram, plot_histogram_logarithmic, plot_boxplot, create_histograms_for_mlp_blocks,
# hooks
from utils import create_hook


def get_config():
    parser = get_public_config()

    results_dir = SCRIPT_DIR.parents / "results"
    plots_dir = f"{results_dir}/plots"

    # Logger
    # log_dir = "{}/{}/{}/seq_len_{}_pred_len_{}_bs_{}/".format(
    #     base_dir,
    #     args.model_name + "_without_sam",
    #     args.dataset,
    #     args.seq_len,
    #     args.horizon,
    #     args.batch_size,
    # )
    log_dir = results_dir

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger(log_dir, __name__, "record_s{}.log".format(args.seed))
    logger.info(args)

    return args, log_dir, logger, plots_dir

def main():
    args, log_dir, logger = get_config()
    set_seed(args.seed)

    # Set up results directory based on model
    model_name = args.llm_model
    results_dir = f"{model_name}_results"
    plots_dir = f"{results_dir}_results"

    # Create directories if they don't exist
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    Path(plots_dir).mkdir(parents=True, exist_ok=True)

    # Prepare input text
    tokenizer,model = get_llm_model(device="cuda")
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
    # top5_labels = [labels[class_idx] for class_idx in top5_classes]
    # labels = ["label1", "label2", ...]  # Define your labels

    # Print the top 5 predictions
    for i, class_idx in enumerate(top5_classes):
        print(f'Top {i+1} predicted class index: {class_idx}')  # Print class indices

    # Backward pass for the highest probability class (if needed)
    # If you want to compute gradients, you need to ensure that requires_grad=True
    logits = logits.squeeze(0)
    logits.requires_grad_()
    logits[0, top5_classes[0][0]].backward()

