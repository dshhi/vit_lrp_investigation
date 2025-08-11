import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import argparse
import pdb
from itertools import combinations

# Custom dataset class for on-the-fly sampling
class SyntheticDataset(Dataset):
    def __init__(self, num_classes, feature_dim, class_variance, samples_per_class):
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.class_variance = class_variance
        self.samples_per_class = samples_per_class
        self.total_samples = num_classes * samples_per_class
        
        # Generate fixed class means
        self.class_means = np.random.normal(0, 1, (num_classes, feature_dim))
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        # Determine which class this sample belongs to
        class_idx = idx // self.samples_per_class
        # Generate sample on-the-fly
        sample = np.random.normal(self.class_means[class_idx], self.class_variance)
        return torch.FloatTensor(sample), class_idx

def parse_args():
    parser = argparse.ArgumentParser(description='Synthetic Linear Classifier Experiment')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
    parser.add_argument('--feature_dim', type=int, default=2046, help='Number of features per sample')
    parser.add_argument('--min_classes', type=int, default=1000, help='Minimum number of classes')
    parser.add_argument('--max_classes', type=int, default=10000, help='Maximum number of classes')
    parser.add_argument('--class_step', type=int, default=1000, help='Step size for class range')
    parser.add_argument('--variances', nargs='+', type=float, default=[0.1, 0.5, 1.0, 2.0, 5.0], 
                        help='List of class variances to test')
    parser.add_argument('--samples_per_class_train', type=int, default=5, 
                        help='Number of samples per class for training')
    parser.add_argument('--samples_per_class_test', type=int, default=2, 
                        help='Number of samples per class for testing')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    
    return parser.parse_args()

def calculate_pairwise_angles(weight_matrix):
    """Calculate angles between all pairs of weight vectors"""
    num_classes = weight_matrix.shape[0]
    angles = []
    
    # Calculate angle between every pair of class weight vectors
    for i, j in combinations(range(num_classes), 2):
        vec1 = weight_matrix[i]
        vec2 = weight_matrix[j]
        
        # Calculate cosine similarity
        cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        
        # Clamp to avoid numerical errors
        cos_sim = np.clip(cos_sim, -1.0, 1.0)
        
        # Calculate angle in radians
        angle = np.arccos(cos_sim)
        angles.append(angle)
    
    return np.array(angles)

def main():
    args = parse_args()
    
    # Setup device
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Parameters from args
    feature_dim = args.feature_dim
    num_classes_range = range(args.min_classes, args.max_classes + 1, args.class_step)
    # num_classes_range = range(9000, args.max_classes + 1, 1000)
    class_variances = args.variances
    samples_per_class_train = args.samples_per_class_train
    samples_per_class_test = args.samples_per_class_test
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.lr
    
    results = []
    
    # Main experiment loop with progress bar
    total_experiments = len(num_classes_range) * len(class_variances)
    experiment_pbar = tqdm(total=total_experiments, 
                          desc="Overall Experiment Progress", position=0)
    
    for num_classes in num_classes_range:
        for class_variance in class_variances:
            tqdm.write(f"Processing: Classes={num_classes}, Variance={class_variance}")
            
            # # Create training dataset (samples generated on-the-fly)
            # train_dataset = SyntheticDataset(num_classes, feature_dim, class_variance, samples_per_class_train)
            # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            #
            # # Create small test set for validation error measurement
            # test_dataset = SyntheticDataset(num_classes, feature_dim, class_variance, samples_per_class_test)
            # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            # Create base dataset with shared class means
            base_dataset = SyntheticDataset(num_classes, feature_dim, class_variance, samples_per_class_train)

            # Create training dataset 
            train_dataset = base_dataset
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            # Create test dataset with same class means
            test_dataset = SyntheticDataset(num_classes, feature_dim, class_variance, samples_per_class_test)
            test_dataset.class_means = base_dataset.class_means  # Share the same class means
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            # Define linear classifier and move to device
            model = nn.Linear(feature_dim, num_classes).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)
            
            # Training loop with progress bar
            model.train()
            epoch_pbar = tqdm(range(epochs), desc=f"Training C={num_classes}, Var={class_variance}", 
                             leave=False, position=1)
            
            for epoch in epoch_pbar:
                # Batch training progress bar
                batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Batches", 
                                 leave=False, position=2)
                
                for batch_x, batch_y in batch_pbar:
                    # Move data to device
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    # Update batch progress bar with current loss
                    batch_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
                batch_pbar.close()
            
            epoch_pbar.close()
            
            # Evaluate on test set (validation error) with progress bar
            model.eval()
            correct = 0
            total = 0
            
            eval_pbar = tqdm(test_loader, desc="Evaluating Model", leave=False, position=1)
            with torch.no_grad():
                for batch_x, batch_y in eval_pbar:
                    # Move data to device
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    
                    outputs = model(batch_x)
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
                    
                    # Update evaluation progress bar
                    eval_pbar.set_postfix({'current_acc': f'{correct/total:.4f}'})
            
            eval_pbar.close()
            accuracy = correct / total
            
            # Analyze weight matrix variance (move back to CPU for numpy operations)
            weights = model.weight.data.cpu().numpy()  # Shape: (num_classes, feature_dim)
            # Calculate all pairwise angles
            pairwise_angles = calculate_pairwise_angles(weights)

            # Convert to degrees for easier interpretation (optional)
            angles_degrees = np.degrees(pairwise_angles)

            # Calculate statistics
            mean_angle = np.mean(angles_degrees)
            variance_angle = np.var(angles_degrees)
            results.append({
                'num_classes': num_classes,
                'class_variance': class_variance,
                'mean_angle': mean_angle,
                'variance_angle': variance_angle,
                'validation_accuracy': accuracy
            })
            
            tqdm.write(f"Classes: {num_classes}, Variance: {class_variance:.1f} => "
                    f"Angle Mean: {mean_angle:.6f}, Angle Variance: {variance_angle:.6f}, Val Acc: {accuracy:.4f}")            
            # Update main experiment progress bar
            experiment_pbar.update(1)
    
    experiment_pbar.close()
    
    # Display results
    import pandas as pd
    print("\nFinal Results:")
    df = pd.DataFrame(results)
    print(df)
    
    # Optionally save results
    df.to_csv('experiment_results.csv', index=False)
    print("\nResults saved to experiment_results.csv")

if __name__ == "__main__":
    main()
