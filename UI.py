import torch
from torchvision import transforms
from PIL import Image
import logging
from pathlib import Path
import streamlit as st
import base64
from typing import Optional, Tuple
import os
import g4f  # Import the g4f module (or adapt if needed)
from g4f.client import Client
import asyncio  # Import asyncio




# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class XRayAnalyzer:
    CLASSES = ["Normal", "Pneumonia"]
    MODEL_PATH = "Model/model.pth"

    def __init__(self):
        self.model = self._load_model()
        self.transform = self._get_transforms()
        self.client = Client()

    def _load_model(self) -> torch.nn.Module:
        """Load and initialize the PyTorch model."""
        try:
            # Ensure the model path exists
            model_path = Path(self.MODEL_PATH)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found at {model_path}")

            model = torch.load(model_path, map_location=torch.device("cpu"))
            model.eval()
            logger.info("Model loaded successfully")
            return model

        except Exception as e:
            logger.error(f"Model loading error: {str(e)}")

    @staticmethod
    def _get_transforms():
        """Define image transformations."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def predict_with_torch(self, image_path: str) -> Tuple[str, float]:
        """
        Predict using PyTorch model.
        Returns prediction and confidence score.
        """
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.transform(image).unsqueeze(0)

            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)

            return self.CLASSES[predicted.item()], confidence.item()
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise RuntimeError("Error during model prediction.")

    
def main():
    st.title("Pneumonia Detection Chatbot")

    # Greet the user
    st.write("Welcome to the Pneumonia Detection Chatbot! Please upload a chest X-ray image.")

    st.markdown('''#### **⚠️ WARNING! DISCLAIMER! ⚠️**
        This tool is for **demonstration purposes only** and should
    **not** be used for medical decision-making. Consult with a qualified 
    healthcare provider for any medical concerns regarding melanoma or skin lesions.''')


    # Create temp directory if it doesn't exist
    temp_dir = "temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # File upload for chest X-ray image
    uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Save uploaded file to local disk temporarily in the 'temp' directory
        image_path = os.path.join(temp_dir, uploaded_file.name)
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.markdown('----')
       
        col1, col2 = st.columns(2)  # Create two columns

        # Display the image in the first column
        with col1:
            st.image(image_path, caption="Uploaded Image")
            st.toast("Image uploaded successfully!")

        # Create an instance of XRayAnalyzer and analyze the image
        analyzer = XRayAnalyzer()

        with col2:
            with st.spinner("Analyzing the X-ray..."):
                try:
                    
                    prediction, confidence = analyzer.predict_with_torch(image_path)

                    # Display the response (prediction and analysis)
                    if prediction and confidence:
                        st.metric(label="Prediction", value=prediction)  
                        st.metric(label="Confidence", value=f"{confidence*100:.2f} %") 
                    else:
                        st.write("An error occurred. Could not make a prediction.")

                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")

    else:
        st.warning("Please upload an image to get started.")

if __name__ == "__main__":
    main()
