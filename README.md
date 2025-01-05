# Pneumonia Detection and Analysis System

## Overview

[](Images/1.png)

This project implements an AI-powered Pneumonia Detection and Analysis System using a Convolutional Neural Network (CNN). The system is designed to detect pneumonia from chest X-ray images, providing predictions and detailed analysis through an interactive user interface built with **Streamlit**.

## Features

- **Pneumonia Detection**: Utilizes a custom-trained CNN model to detect pneumonia in chest X-ray images with high accuracy.
- **Interactive User Interface**: Built using **Streamlit** for a seamless and user-friendly experience.
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
├── UI.py                         # Interactive UI using Streamlit
├── pipeline.py                   # Pipeline for training and inference                  
├── Dockerfile                    # Docker configuration for containerized app
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

## Installation

### Prerequisites

- Python 3.8+
- PyTorch
- Streamlit

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

### Running the Application

- To run the application locally with the **Streamlit** UI:

  ```bash
  streamlit run UI.py
  ```

[](Images/2.png)

- To run the pipeline for training the model:
  
  ```bash
  python pipeline.py
  ```

## Model Details

[](Images/3.png)

The system employs the `PneumoniaCNN` architecture, a custom-built CNN with three convolutional blocks and a fully connected layer. The model is trained using chest X-ray images, and both custom and pretrained models (such as ResNet) are available for comparison.

[](Images/4.png)

- **Custom Model Accuracy**:
  - Training Accuracy: 85%
  - Testing Accuracy: 89%

- **Pretrained Model Accuracy**:
  - Testing Accuracy: 90%

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
- Streamlit: For creating the interactive UI.

---
