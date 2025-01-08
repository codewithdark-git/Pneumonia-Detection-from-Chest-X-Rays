import logging
from CNN import PneumoniaCNN
from data_loader import load_dataset
from train import PneumoniaTrainer
import torch

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='PneumoniaCNN.log',
    filemode='a'  # 'a' for append mode instead of 'w' for write
)


# Set random seed for reproducibility
torch.manual_seed(42)


# Define dataset directory and parameters
main_dir = 'chest_xray'  # Path to chest X-ray dataset
batch_size = 16
epochs = 5
learning_rate = 0.001
img_size = 224
save_model_path = 'Model/model.pth'

# Load dataset
print("Loading dataset...")
logging.info("Loading dataset...")
train_loader, val_loader, test_loader = load_dataset(main_dir, batch_size=batch_size)
print("Dataset loaded successfully!")

# Initialize the CNN model
print("Initializing model...")
logging.info("Initializing model...")
model = PneumoniaCNN(pretrained=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Device: {device}")
model = model.to(device)

# Initialize the trainer class
print("Preparing training pipeline...")
logging.info("Preparing training pipeline...")

trainer = PneumoniaTrainer(model=model,
                            img_size=224, 
                            batch_size=32,
                            epochs=epochs,
                            learning_rate=learning_rate, 
                            model_name="PneumoniaCNN")


# Train and evaluate
print("Training model...")
trainer.train_model(train_loader)

# Evaluate model
print("Evaluating model...")
trainer.evaluate_accuracy(train_loader, dataset_name="Train")

print("Evaluating model...")
trainer.evaluate_accuracy(test_loader, dataset_name="Test")


# save model
print("Save model...")
trainer.save_model()

# # Visualize training and evaluation metrics
# print("Plotting training and validation metrics...")
# logging.info("Plotting training and validation metrics...")
# trainer.plot_loss_accuracy()

# Generate confusion matrix
print("Generating confusion matrix...")
logging.info("Generating confusion matrix...")
trainer.plot_confusion_matrix(test_loader, classes=['Normal', 'Pneumonia'])

# Display model architecture and save summary
print("Saving model architecture and summary...")
logging.info("Saving model architecture and summary...")
trainer.archit()

print("\nPipeline completed successfully!")
logging.info("Pipeline completed successfully!")
