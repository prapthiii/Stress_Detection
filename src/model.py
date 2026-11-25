from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import numpy as np
import joblib
import os

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
    
    class StressCNN(nn.Module):
        """1D CNN Architecture for Speech Emotion/Stress Recognition"""
        def __init__(self, input_dim, num_classes=3):
            super(StressCNN, self).__init__()
            self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
            self.pool = nn.MaxPool1d(2)
            self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
            # Adjust linear layer depending on flattened input size
            self.fc1 = nn.Linear(32 * (input_dim // 4), 64)
            self.fc2 = nn.Linear(64, num_classes)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x
except ImportError:
    TORCH_AVAILABLE = False

class StressModel:
    def __init__(self, model_type='svm'):
        self.model_type = model_type
        if model_type == 'svm':
            self.model = SVC(probability=True)  # Enable probability for confidence
        elif model_type == 'rf':
            self.model = RandomForestClassifier()
        elif model_type == 'cnn' and TORCH_AVAILABLE:
            # Using 122 as default input dim (example)
            self.model = StressCNN(input_dim=122, num_classes=3)
        else:
            self.model = SVC(probability=True)
        self.model_path = 'models/stress_model.pkl'
        self.torch_model_path = 'models/stress_cnn.pth'


    def train(self, X, y):
        """Train the model with cross-validation."""
        if self.model_type == 'cnn' and TORCH_AVAILABLE:
            print("Training deep learning CNN...")
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            
            # Simple dummy single-pass execution for stub
            X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1) # Add channel dim for 1D CNN
            y_tensor = torch.tensor(y, dtype=torch.long)
            
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            
            acc = (outputs.argmax(1) == y_tensor).float().mean().item()
            os.makedirs('models', exist_ok=True)
            torch.save(self.model.state_dict(), self.torch_model_path)
            print(f"Training accuracy (CNN): {acc}")
            return acc
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            self.model.fit(X_train, y_train)
            pred = self.model.predict(X_test)
            acc = accuracy_score(y_test, pred)
            print(f"Training accuracy: {acc}")

            # Cross-validation
            cv_scores = cross_val_score(self.model, X, y, cv=5)
            print(f"Cross-validation accuracy: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")

            # Save model
            os.makedirs('models', exist_ok=True)
            joblib.dump(self.model, self.model_path)
            return acc

    def load_model(self):
        """Load pre-trained model."""
        if self.model_type == 'cnn' and TORCH_AVAILABLE:
            if os.path.exists(self.torch_model_path):
                self.model.load_state_dict(torch.load(self.torch_model_path))
                self.model.eval()
                return True
            return False
        else:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                return True
            return False

    def predict(self, X):
        """Predict stress level."""
        if self.model_type == 'cnn' and TORCH_AVAILABLE:
            with torch.no_grad():
                X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
                outputs = self.model(X_tensor)
                return outputs.argmax(1).numpy()
        return self.model.predict(X)

    def predict_proba(self, X):
        """Predict probabilities for confidence."""
        if self.model_type == 'cnn' and TORCH_AVAILABLE:
            with torch.no_grad():
                X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
                outputs = self.model(X_tensor)
                return torch.softmax(outputs, dim=1).numpy()
        return self.model.predict_proba(X)

# Example usage
if __name__ == "__main__":
    model = StressModel()
    # Dummy data
    X = np.random.rand(100, 122)
    y = np.random.randint(0, 3, 100)  # 0: low, 1: moderate, 2: high
    model.train(X, y)
    pred = model.predict(X[:5])
    print("Predictions:", pred)