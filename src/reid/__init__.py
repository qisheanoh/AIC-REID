import cv2
import numpy as np
from src.reid.extractor import ReidExtractor

if __name__ == "__main__":
    # Load and prepare sample crop
    img = cv2.imread("sample.jpg")
    if img is None:
        raise FileNotFoundError("Make sure 'sample.jpg' exists in the working directory.")
        
    crop = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (128, 256))

    extractor = ReidExtractor(model_name="osnet_x1_0", device="cpu")
    features = extractor([crop])  # expects list of np.ndarray

    print("Feature shape:", features.shape)  # Should be (1, 512)
