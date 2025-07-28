import argparse


def get_public_config():
    parser = argparse.ArgumentParser()
    # Hardware
    parser.add_argument("--device", type=str, default="cuda:0")
    # parser.add_argument("--device", type=str, default="cuda")

    # Model
    parser.add_argument('--llm_model', choices=['Qwen3-0.6B'], required=False, 
                        help='Select the llm model to use.', default = 'Qwen3-0.6B')
    parser.add_argument('--vit_model', type=str, choices=['vitb16', 'vitl16'], 
                        required=False, help='Specify the vit model to use: vitb16 or vitl16', default='vitb16')
    parser.add_argument("--seed", type=int, default=1)

    # Dataset
    #

    return parser
