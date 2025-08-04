import argparse


def get_public_config():
    parser = argparse.ArgumentParser()
    # Hardware
    parser.add_argument("--device", type=str, default="cuda:0")
    # parser.add_argument("--device", type=str, default="cuda")

    # Model
    parser.add_argument('--model', type=str, choices=['qwen3-0.6B','qwen2-7B','qwen2-0.5B', 'gemma-3-12b-it', 'gemma-3-4b-it','vitb16', 'vitl16' ], required=True, 
                        help='Select the model to use in lowercase.', default = 'vitb16')
    parser.add_argument("--seed", type=int, default=1)

    # Dataset
    #

    return parser
