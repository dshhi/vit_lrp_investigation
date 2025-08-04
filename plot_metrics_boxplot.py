import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from io import StringIO
import matplotlib.cm as cm

def parse_args():
    parser = argparse.ArgumentParser(description="Create boxplots from model results.")
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Model name(s) or 'all' to include all models"
    )
    parser.add_argument(
        "--type",
        choices=["vision", "language", "both"],
        default="both",
        help="Type of models to process (vision, language, or both)"
    )
    parser.add_argument(
        "--exclude-outliers",
        action="store_true",
        help="Exclude outliers from boxplots"
    )
    return parser.parse_args()

def load_model_data(model_path):
    """Load data from metrics.md file."""
    md_file = model_path / "metrics" / "metrics.md"
    if not md_file.exists():
        raise FileNotFoundError(f"metrics.md not found in {md_file}")
    
    # Read the markdown table, skipping the separator row
    with open(md_file, 'r') as f:
        lines = f.readlines()
    
    # Remove the separator line (usually the second line)
    cleaned_lines = [line for line in lines if not line.strip().startswith('|-')]
    
    # Join the lines back and read with pandas
    content = ''.join(cleaned_lines)
    df = pd.read_csv(StringIO(content), sep="|", header=0, skipinitialspace=True)
    
    # Clean column names and remove empty columns
    df.columns = [col.strip() for col in df.columns]
    df = df.dropna(axis=1, how='all')
    df = df.loc[:, df.columns != '']
    
    # Convert numeric columns safely
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            pass
    
    return df

def get_model_directories(base_dir, model_type, models_arg):
    """Get list of model directories based on type and models argument."""
    model_dirs = []
    
    if model_type == "both":
        search_dirs = [base_dir / "vision", base_dir / "language"]
    else:
        search_dirs = [base_dir / model_type]
    
    if "all" in models_arg:
        for search_dir in search_dirs:
            if search_dir.exists():
                model_dirs.extend([search_dir / d.name for d in search_dir.iterdir() if d.is_dir()])
    else:
        # Look for specific models in both directories
        for model_name in models_arg:
            found = False
            for search_dir in search_dirs:
                if search_dir.exists():
                    model_path = search_dir / model_name
                    if model_path.exists():
                        model_dirs.append(model_path)
                        found = True
            if not found:
                print(f"Warning: Model '{model_name}' not found. Skipping...")
    
    return model_dirs

def get_model_names_for_filename(model_paths, model_type, models_arg):
    """Generate appropriate model names for filename."""
    if "all" in models_arg:
        if model_type == "vision":
            return "all_vision_models"
        elif model_type == "language":
            return "all_language_models"
        else:  # both
            return "all_models"
    else:
        # Use actual model names
        names = [path.name for path in model_paths]
        return "_".join(sorted(names))

def main():
    args = parse_args()
    
    # Get base directory
    base_dir = Path("results")
    if not base_dir.exists():
        raise FileNotFoundError("Results directory does not exist")

    # Get model directories
    try:
        model_paths = get_model_directories(base_dir, args.type, args.models)
        if not model_paths:
            print("No valid models found.")
            return
    except Exception as e:
        print(f"Error getting model directories: {e}")
        return

    # Determine model names for filename
    model_names_for_file = get_model_names_for_filename(model_paths, args.type, args.models)

    # Create output directory
    plots_base_dir = base_dir / "plots" / args.type
    plots_base_dir.mkdir(parents=True, exist_ok=True)

    # Get MLP blocks from first model (assuming same for all)
    first_model_df = load_model_data(model_paths[0])
    if 'MLP Block' not in first_model_df.columns:
        raise ValueError("Missing 'MLP Block' column in metrics data")
    
    mlp_blocks = first_model_df['MLP Block'].tolist()
    
    # Create one plot per MLP block
    for i, mlp_block in enumerate(mlp_blocks):
        plt.figure(figsize=(12, 8))
        
        # Collect data for all models at this MLP block
        all_model_data = []
        model_labels = []
        
        # Generate colors for each model
        colors = cm.tab10(np.linspace(0, 1, len(model_paths)))
        
        for model_idx, model_path in enumerate(model_paths):
            try:
                df = load_model_data(model_path)
                
                # Collect all metric values for this MLP block (excluding MLP Block column)
                metric_values = []
                for col in df.columns:
                    if col != 'MLP Block' and i < len(df[col]):
                        val = df[col].iloc[i]
                        if pd.notna(val):  # Skip NaN values
                            metric_values.append(val)
                
                if metric_values:
                    all_model_data.append(metric_values)
                    model_labels.append(f"{model_path.parent.name}/{model_path.name}")
                else:
                    print(f"Warning: No valid data for {model_path.name} at MLP block {mlp_block}")
                    
            except Exception as e:
                print(f"Error processing model '{model_path.name}': {e}")
                continue
        
        if not all_model_data:
            print(f"No valid data found for MLP block {mlp_block}")
            plt.close()
            continue
            
        # Create boxplot with proper parameter for newer matplotlib
        boxprops = dict(linewidth=1.5)
        whiskerprops = dict(linewidth=1.5)
        capprops = dict(linewidth=1.5)
        medianprops = dict(linewidth=2, color='red')
        
        # Create positions for boxes
        positions = range(1, len(all_model_data) + 1)
        
        # Use tick_labels instead of labels for newer matplotlib versions
        bp = plt.boxplot(
            all_model_data,
            positions=positions,
            showfliers=not args.exclude_outliers,
            boxprops=boxprops,
            whiskerprops=whiskerprops,
            capprops=capprops,
            medianprops=medianprops
        )
        
        # Set labels separately to avoid deprecation warning
        plt.xticks(positions, model_labels, rotation=45, ha='right')
        
        # Color each boxplot differently (fix the previous error)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_gapcolor(color)
            patch.set_alpha(0.7)
        
        # Configure plot
        plt.ylabel('Metric Values')
        plt.title(f'Metric Value Distribution at MLP Block {mlp_block}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        filename = f"box_plot_mlp_{mlp_block}_{model_names_for_file}.png"
        save_path = plots_base_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved boxplot: {save_path}")

if __name__ == "__main__":
    main()
