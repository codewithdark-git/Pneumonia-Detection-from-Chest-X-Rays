
# Pneumonia Detection and Analysis System

## Overview

This project implements an AI-powered Pneumonia Detection and Analysis System using a Convolutional Neural Network (CNN). The system is designed to detect pneumonia from chest X-ray images, providing detailed analysis and predictions using Groq Vision for medical image interpretation. The system also offers a user-friendly conversational interface using Chainlit for seamless interaction.

## Features

- **Pneumonia Detection**: Utilizes a custom-trained CNN model to detect pneumonia in chest X-ray images with high accuracy.
- **Detailed Image Analysis**: Groq Vision provides advanced image analysis and text-based explanations.
- **Conversational UI**: Built using Chainlit for a smooth user experience.
- **Security**: Implements AES-256 encryption for image data to comply with HIPAA regulations.

## Project Structure

```
pneumonia-detection/
├── chest_xray/                   # Directory for dataset and image data          
│   ├── train/
│   ├── val/                   
│   └── test/
├── Images                        # Directory for plotting images                   
├── models/                       # Directory for trained models
│   ├── pneumonia_cnn.pth         # Custom-trained CNN model
│   └── pretrained_model.pth      # Pre-trained model (e.g., ResNet)
├── pneumoniaCNN.ipynb                          
├── data_loader.py            
├── CNN.py                  
├── train.py                                        
├── UI.py                         # Conversational UI using Chainlit
├── pipeline.py                   # Pipeline for training and inference                  
├── Dockerfile                    # Docker configuration for containerized app
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

## Installation

### Prerequisites

- Python 3.8+
- PyTorch
- Groq Vision SDK
- Chainlit

### Setup Instructions

1. Clone the repository:

   ```bash
   git clone https://github.com/codewithdark-git/Pneumonia-Detection-from-Chest-X-Rays.git
   cd Pneumonia-Detection-from-Chest-X-Rays
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up Groq Vision API credentials and Chainlit configuration (refer to their documentation for setup).

### Running the Application

- To run the application locally with UI integration:

  ```bash
  Chainlit run UI.py
  ```
- To run the pipeline for Training Model:
  
  ```bash
  python pipeline.py
  ```

## Model Details

The system employs the `PneumoniaCNN` architecture, a custom-built CNN with three convolutional blocks and a fully connected layer. The model is trained using chest X-ray images, and both custom and pretrained models (such as ResNet) are available for comparison.

- **Custom Model Accuracy**:
  - Training Accuracy: 85%
  - Testing Accuracy: 89%

- **Pretrained Model Accuracy**:
  - Testing Accuracy: 90%

## API Integration

### Groq Vision

The system integrates with Groq Vision to provide detailed text-based explanations of the analysis. The analysis includes highlighting areas of the X-ray that show potential signs of pneumonia, improving the interpretability of the AI system for medical professionals.

## Security and Privacy

- **Data Encryption**: All uploaded X-ray images are encrypted using AES-256 encryption.
- **HIPAA Compliance**: The system adheres to HIPAA regulations for secure handling of medical data.

## Future Enhancements

- **Mobile App**: Develop a mobile version of the app for easier access in healthcare environments.
- **Batch Processing**: Implement a system for batch processing of multiple X-ray images.
- **Model Optimization**: Experiment with more advanced CNN architectures to improve accuracy and inference speed.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- PyTorch: For building the deep learning models.
- Groq Vision: For advanced image analysis.
- Chainlit: For creating the conversational UI.

--- 
