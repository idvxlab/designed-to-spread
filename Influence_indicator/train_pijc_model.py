#!/usr/bin/env python3
"""
Train PIJC (Probability of Information Cascade) prediction model
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import random
import os
import numpy as np


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, folder_path):
        self.in_user_feature = None
        self.out_user_feature = None
        self.message_feature = None
        self.y = []

        pt_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.pt')]
        for pt_file in pt_files:
            data = torch.load(pt_file) 
            if self.in_user_feature is None:
                self.in_user_feature = data['in_user_feature']
                self.out_user_feature = data['out_user_feature']
                self.message_feature = data['message_feature']
            else:
                self.in_user_feature = torch.cat((self.in_user_feature, data['in_user_feature']), dim=0)
                self.out_user_feature = torch.cat((self.out_user_feature, data['out_user_feature']), dim=0)
                self.message_feature = torch.cat((self.message_feature, data['message_feature']), dim=0)
            
            self.y.extend(data['y']) 

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (self.in_user_feature[idx], self.out_user_feature[idx], self.message_feature[idx], self.y[idx])

# Model definition
class MultiInputNet(nn.Module):
    def __init__(self, input_dim=512, num_classes=2, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)):
        super(MultiInputNet, self).__init__()

        self.mean = nn.Parameter(mean, requires_grad=False)
        self.std = nn.Parameter(std, requires_grad=False)
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def normalize(self, x, mean, std):
        mean = mean.to(x.device)
        std = std.to(x.device)
        return (x - mean) / std

    def forward(self, a1, a2, c):
        a1 = self.normalize(a1, self.mean[0], self.std[0])
        a2 = self.normalize(a2, self.mean[1], self.std[1])
        c = self.normalize(c, self.mean[2], self.std[2])
        a1_features = self.feature_extractor(a1)
        a2_features = self.feature_extractor(a2)
        c_features = self.feature_extractor(c)
        combined_features = torch.cat((a1_features, a2_features, c_features), dim=1)
        output = self.classifier(combined_features)
        return output


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU usage
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_mean_std(loader):
    """Calculate mean and std of training data"""
    all_a1, all_a2, all_c = [], [], []

    for a1, a2, c, _ in loader:
        all_a1.append(a1)
        all_a2.append(a2)
        all_c.append(c)

    all_a1 = torch.cat(all_a1, dim=0)
    all_a2 = torch.cat(all_a2, dim=0)
    all_c = torch.cat(all_c, dim=0)

    mean_a1, std_a1 = all_a1.mean(dim=0), all_a1.std(dim=0)
    mean_a2, std_a2 = all_a2.mean(dim=0), all_a2.std(dim=0)
    mean_c, std_c = all_c.mean(dim=0), all_c.std(dim=0)

    mean = torch.stack((mean_a1, mean_a2, mean_c))
    std = torch.stack((std_a1, std_a2, std_c))

    return mean, std


def train(model, train_loader, criterion, optimizer, device):
    """Training function"""
    model.train()
    total_loss = 0
    print('begin training....')
    for a1, a2, c, targets in train_loader:
        a1, a2, c, targets = a1.to(device).float(), a2.to(device).float(), c.to(device).float(), targets.to(device).long()
        optimizer.zero_grad()
        outputs = model(a1, a2, c)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def evaluate(model, loader, criterion, device):
    """Evaluation function"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for a1, a2, c, targets in loader:
            a1, a2, c, targets = a1.to(device).float(), a2.to(device).float(), c.to(device).float(), targets.to(device).long()
            outputs = model(a1, a2, c)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[:, 1]

            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())

    accuracy = accuracy_score(all_targets, all_predictions)
    avg_loss = total_loss / len(loader)

    # Generate classification report
    report = classification_report(all_targets, all_predictions, target_names=['Class 0', 'Class 1'])

    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(all_targets, all_probs)
    roc_auc = auc(fpr, tpr)

    return avg_loss, accuracy, report, fpr, tpr, roc_auc


def main():
    """Main function"""
    # Get script directory and use relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(os.path.dirname(script_dir), 'Data', 'outputs')
    dataset_folder_train = os.path.join(base_path, 'outputs_train')
    dataset_folder_test = os.path.join(base_path, 'outputs_test')
    
    # Create model save directory
    model_dir = os.path.join(base_path, 'model')
    os.makedirs(model_dir, exist_ok=True)
    
    # Set random seed
    seed = 77
    set_seed(seed)

    # Load training and test datasets
    print("Loading datasets...")
    pre_dataset_1 = CustomDataset(dataset_folder_train)
    train_val_dataset = random.sample(list(pre_dataset_1), len(pre_dataset_1))

    pre_dataset_2 = CustomDataset(dataset_folder_test)
    test_dataset = random.sample(list(pre_dataset_2), len(pre_dataset_2))

    train_loader = DataLoader(train_val_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # Calculate mean and std of training data
    print("Calculating mean and std...")
    train_mean, train_std = calculate_mean_std(train_loader)

    # Model, loss function and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = MultiInputNet(input_dim=512, num_classes=2, mean=train_mean, std=train_std).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    print("Starting training...")
    num_epochs = 100
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")

        # Save model
        if (epoch + 1) % 10 == 0:
            model_path = os.path.join(model_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to: {model_path}")

    # Evaluate model on test set
    print("\nEvaluating on test set...")
    test_loss, test_accuracy, test_report, test_fpr, test_tpr, test_roc_auc = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    print(test_report)

    # Plot and save ROC curve
    print("Saving ROC curve...")
    plt.figure()
    plt.plot(test_fpr, test_tpr, color='darkorange', lw=2, label='ROC curve (area = %0.3f)' % test_roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic ')
    plt.legend(loc="lower right")
    roc_curve_path = os.path.join(model_dir, "test_roc_curve_all.png")
    plt.savefig(roc_curve_path)
    plt.close()
    print(f"ROC curve saved to: {roc_curve_path}")
    print("\n All done!")


if __name__ == "__main__":
    main()

