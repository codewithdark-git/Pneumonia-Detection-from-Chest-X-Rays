import torch
from torchvision import transforms
from PIL import Image
import logging
from pathlib import Path
import chainlit as cl
from groq import Groq
import base64
from typing import Optional, Tuple
from CNN import PneumoniaCNN

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
        self.groq_model = Groq()

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
            logger.error('')

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

    async def analyze_with_groq(self, image_path: str, prompt: Optional[str] = None) -> str:
        """
        Analyze image using Groq Vision.
        
        Args:
            image_path: Path to the image file
            prompt: Optional user prompt to guide the analysis
        """
        try:
            # Encode image to base64
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')

            # Prepare message with or without custom prompt
            message_text = prompt if prompt else "Analyze this chest X-ray image in detail. Describe any visible abnormalities or signs of pneumonia."
            
            # Create chat completion request
            chat_completion = self.groq_model.chat.completions.create(
            messages=[
                {
                "role": "user",
                "content": [
                    {"type": "text", "text": message_text},
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                    },
                ],
                }
            ],
            model="llama-3.2-11b-vision-preview",
            )
            
            return chat_completion.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Groq analysis error: {str(e)}")
            return "Error in generating detailed analysis using Groq Vision."

    async def generate_combined_response(self, image_path: str, user_prompt: Optional[str] = None) -> str:
        """
        Generate a combined response using PyTorch and Groq Vision models.
        """
        try:
            # Step 1: Get prediction from PyTorch model
            prediction, confidence = self.predict_with_torch(image_path)

            # Step 2: Get detailed analysis from Groq Vision
            groq_insights = await self.analyze_with_groq(image_path, prompt=user_prompt)

            # Step 3: Combine results
            response = (
                f"### Prediction\n"
                f"**The X-ray indicates:** {prediction} (Confidence: {confidence:.2%})\n\n"
                f"### Detailed Analysis\n"
                f"{groq_insights}\n\n"
                f"### Disclaimer\n"
                f"⚠️ This is an AI-assisted analysis. Please consult a medical professional for accurate diagnosis."
            )
            return response
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "An error occurred during analysis. Please try again."

@cl.on_chat_start
async def upload_image_ui():
    """
    Chainlit UI for image upload and analysis.
    """
    analyzer = XRayAnalyzer()

    try:
        # Step 1: File upload
        uploaded_file = await cl.FileUpload(
            label="Upload your chest X-ray image",
            accept=["image/jpeg", "image/png"]
        )
        if not uploaded_file:
            await cl.Message(content="❌ No file uploaded. Please upload an image.").send()
            return
        
        await cl.Message('''Welcome to the Pneumonia Detection Chatbot!
                         ''').send()

        image_path = uploaded_file.path
        if not Path(image_path).exists():
            await cl.Message(content="❌ Invalid file. Please upload a valid image.").send()
            return

        await cl.Message(content="✅ Image uploaded successfully!").send()

        # Step 2: Optional user text input
        user_prompt = await cl.Text(
            label="Additional context or questions about the X-ray (optional)",
            placeholder="e.g., Patient history, specific concerns, or questions"
        )

        # Step 3: Analysis
        with cl.Loading("Analyzing the image..."):
            response = await analyzer.generate_combined_response(image_path, user_prompt)

        # Step 4: Display results
        await cl.Message(content=response).send()

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        logger.error(error_message)
        await cl.Message(content=error_message).send()

if __name__ == "__main__":
    logger.info("Starting X-Ray Analysis System")
