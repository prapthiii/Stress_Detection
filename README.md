# Stress Management System

A multimodal stress detection system that analyzes voice and text data to classify stress levels and provide supportive suggestions.

## Features

- **Voice Analysis**: Extracts tone, pitch, energy from audio files
- **Text Analysis**: Sentiment analysis from direct text input or OCR from images
- **Stress Prediction**: 4-level classification (No stress, Mild stress, Moderate stress, High stress)
- **OCR Support**: Extracts text from images (screenshots, messages, emails)
- **Supportive Suggestions**: Generates tailored messages based on stress level

## Sample Data

The project includes synthetic sample audio files:
- `data/raw/stressed.wav`: Simulated stressed voice (higher pitch, more energy)
- `data/raw/not_stressed.wav`: Simulated calm voice (lower pitch, less energy)

## Components

1. **Data Collection**: Loads sample audio or records voice
2. **Data Preprocessing**: Cleans audio and text
3. **Feature Extraction**: Extracts MFCC, pitch, energy from voice; sentiment from text
4. **Feature Fusion**: Combines voice and text features
5. **Machine Learning Model**: SVM for stress prediction
6. **Stress Classification**: Predicts Stressed or Not Stressed
7. **Support Suggestions**: Provides coping strategies if stressed
8. **Evaluation & Feedback**: Measures model performance

## Installation

1. Install Python 3.8+
2. `pip install -r requirements.txt`

## OCR Setup (Optional)
For image text extraction:
1. Download Tesseract OCR from: https://github.com/UB-Mannheim/tesseract/wiki
2. Install it (add to PATH)
3. If needed, set the path in `app.py`: `pytesseract.pytesseract.tesseract_cmd = r'C:\path\to\tesseract.exe'`

## Data

Place raw data in `data/raw/`. Processed data in `data/processed/`.

## Models

Trained models saved in `models/`.

# project