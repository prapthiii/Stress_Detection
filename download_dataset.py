import kaggle
import os

# Authenticate with Kaggle API
kaggle.api.authenticate()

# Download the SAVEE dataset (Surrey Audio-Visual Expressed Emotion)
# This dataset contains audio recordings of emotions including stress-related ones
dataset_slug = 'barelydedicated/savee-database'
download_path = 'data/raw/'

# Create directory if it doesn't exist
os.makedirs(download_path, exist_ok=True)

# Download and unzip the dataset
kaggle.api.dataset_download_files(dataset_slug, path=download_path, unzip=True)

print(f"Dataset downloaded to {download_path}")