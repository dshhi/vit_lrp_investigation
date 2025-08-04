import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from io import StringIO

def parse_args():
    parser = argparse.ArgumentParser(description="Plot metrics from model results.")
    parser.add_argument(
        "--metrics",
        nargs="+",
        required=True,
        help="Metric(s) to plot (e.g., Mean Variance 'Max Angle') or 'all' for all metrics"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Model name(s) or 'all' to include all models"
    )
    parser.add_argument(
        "--type",
        choices=["vision", "language", "both"],
        default="both",
        help="Type of models to process (vision, language, or both)"
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Plot logarithm of metrics"
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
    cleaned_lines = []
    for line in lines:
        if not line.strip().startswith('|-'):
            cleaned_lines.append(line)
    
    # Join the lines back and read with pandas
    content = ''.join(cleaned_lines)
    df = pd.read_csv(StringIO(content), sep="|", header=0, skipinitialspace=True)
    
    # Clean column names and remove empty columns
    df.columns = [col.strip() for col in df.columns]
    df = df.dropna(axis=1, how='all')
    
    # Remove empty columns (from leading/trailing |)
    df = df.loc[:, df.columns != '']
    
    # Convert numeric columns safely
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            # Keep as string if conversion fails
            pass
    
    return df

def get_available_metrics(base_dir, model_type):
    """Extract available metrics from the first found metrics.md file."""
    if model_type == "both":
        search_dirs = [base_dir / "vision", base_dir / "language"]
    else:
        search_dirs = [base_dir / model_type]
    
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for model_dir in search_dir.iterdir():
            if model_dir.is_dir():
                md_file = model_dir / "metrics" / "metrics.md"
                if md_file.exists():
                    try:
                        df = load_model_data(model_dir)
                        # Return all columns except the first one ('MLP Block')
                        available = [col for col in df.columns if col != 'MLP Block']
                        if available:
                            return available
                    except Exception as e:
                        print(f"Warning: Could not parse {md_file}: {e}")
                        continue
    raise FileNotFoundError("No valid metrics.md file found to extract column names")

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
                print(f"Warning: Model '{model_name}' not found in any directory. Skipping...")
    
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
        names = []
        for path in model_paths:
            names.append(path.name)
        return "_".join(sorted(names))

def apply_log_if_needed(data, log_flag):
    """Apply logarithm to data if flag is set, handling negative values."""
    if not log_flag:
        return data
    
    # Handle negative and zero values
    if np.any(data <= 0):
        # Shift data to make all values positive
        min_val = np.min(data)
        if min_val <= 0:
            shifted_data = data + abs(min_val) + 1e-10  # Add small epsilon to avoid log(0)
        else:
            shifted_data = data
        return np.log(shifted_data)
    else:
        return np.log(data)

def main():
    args = parse_args()
    
    # Get base directory
    base_dir = Path("results")
    if not base_dir.exists():
        raise FileNotFoundError("Results directory does not exist")

    # Dynamically get available metrics
    try:
        available_metrics = get_available_metrics(base_dir, args.type)
        print(f"Available metrics: {available_metrics}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Handle "all" metrics option
    if "all" in args.metrics:
        metrics_to_plot = [metric for metric in available_metrics if metric != 'MLP Block']
    else:
        # Validate requested metrics
        invalid_metrics = set(args.metrics) - set(available_metrics)
        if invalid_metrics:
            raise ValueError(f"Invalid metrics: {invalid_metrics}. Available: {available_metrics}")
        metrics_to_plot = args.metrics

    # Get model directories
    if not args.models:
        raise ValueError("Either --models must be specified")
    
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

    # Create output directory based on type
    plots_base_dir = base_dir / "plots" / args.type
    plots_base_dir.mkdir(parents=True, exist_ok=True)

    # Process each metric separately
    for metric in metrics_to_plot:
        plt.figure(figsize=(10, 6))
        has_data = False
        
        for model_path in model_paths:
            try:
                df = load_model_data(model_path)
                if 'MLP Block' not in df.columns:
                    print(f"Warning: 'MLP Block' column missing in {model_path.name}")
                    continue
                
                if metric not in df.columns:
                    print(f"Warning: Metric '{metric}' not found in {model_path.name}")
                    continue
                
                x = df['MLP Block']
                y = df[metric]
                
                # Apply log transformation if requested
                if args.log:
                    y = apply_log_if_needed(y, True)
                
                plt.plot(x, y, marker='o', label=model_path.parent.name + "/" + model_path.name)
                has_data = True
            
            except Exception as e:
                print(f"Error processing model '{model_path.name}': {e}")
                continue
        
        # Only save plot if we have data
        if has_data:
            # Configure plot
            plt.xlabel('MLP Block')
            
            # Set y-axis label
            if args.log:
                ylabel = f'log({metric})'
            else:
                ylabel = metric
            plt.ylabel(ylabel)
            
            # Set title
            if args.log:
                title = f'log({metric}) Across Models'
            else:
                title = f'{metric} Across Models'
            plt.title(title)
            
            plt.legend(loc='upper right')  # Legend to top right
            plt.grid(True)
            
            # Save plot with descriptive filename
            metric_safe_name = metric.replace(" ", "_")
            if args.log:
                filename = f"{metric_safe_name}_Log_{model_names_for_file}.png"
            else:
                filename = f"{metric_safe_name}_{model_names_for_file}.png"
            save_path = plots_base_dir / filename
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved plot: {save_path}")
        else:
            plt.close()
            print(f"No valid data found for metric '{metric}', plot not saved.")

if __name__ == "__main__":
    main()
