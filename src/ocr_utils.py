import re
import numpy as np
from PIL import Image

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

# Initialize EasyOCR reader once
_reader = None

def get_ocr_reader():
    """Lazy-load EasyOCR reader."""
    global _reader
    if _reader is None and EASYOCR_AVAILABLE:
        _reader = easyocr.Reader(['en'])
    return _reader

def clean_text(text):
    """Remove special characters and normalize."""
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower().strip()

def preprocess_image(image):
    """Apply OpenCV preprocessing for better OCR accuracy."""
    if not CV2_AVAILABLE:
        return image
    
    img = np.array(image)
    
    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Apply Gaussian blur to reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Otsu thresholding for better contrast
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return thresh

def extract_text_from_image(image_path):
    """Extract text from image using EasyOCR with OpenCV preprocessing."""
    try:
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Preprocess
        if CV2_AVAILABLE:
            processed_img = preprocess_image(image)
        else:
            processed_img = np.array(image)
        
        text = ""
        
        # Try EasyOCR first
        if EASYOCR_AVAILABLE:
            try:
                reader = get_ocr_reader()
                results = reader.readtext(processed_img)
                text = " ".join([res[1] for res in results])
            except Exception as e:
                print(f"EasyOCR error: {e}")
                # Fallback to Tesseract
                if TESSERACT_AVAILABLE:
                    text = pytesseract.image_to_string(processed_img, config='--oem 3 --psm 6')
                else:
                    return "[ERROR] Both EasyOCR and Tesseract failed. Please install Tesseract OCR."
        else:
            # Fallback to Tesseract if EasyOCR not available
            if TESSERACT_AVAILABLE:
                text = pytesseract.image_to_string(processed_img, config='--oem 3 --psm 6')
            else:
                return "[ERROR] Neither EasyOCR nor Tesseract is available. Install: pip install easyocr"
        
        return clean_text(text)
    except Exception as e:
        print(f"OCR Error: {e}")
        return f"[ERROR] OCR failed: {str(e)}"