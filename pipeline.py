import logging

from cv2 import log
from CNN import PneumoniaCNN
from data_loader import load_dataset
from train import Train
import torch

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='PneumoniaCNN.log',
    filemode='a'  # 'a' for append mode instead of 'w' for write
)

# Enable logging in the cv2 library


# Set random seed for reproducibility
torch.manual_seed(42)


# Define dataset directory and parameters
main_dir = 'chest_xray'  # Path to chest X-ray dataset
batch_size = 16
num_epochs = 10
learning_rate = 0.001
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

trainer = Train(
    model=model,
    model_name=model.__class__.__name__,
    device=device,
    batch_size=batch_size,
    train_loader=train_loader,
    num_epochs=num_epochs,
    learning_rate=learning_rate
)

# Train the model
print("Starting training...")
logging.info("Starting training...")
trainer.trainer()

# Save the trained model
print(f"Saving trained model to '{save_model_path}'...")
logging.info(f"Saving trained model to '{save_model_path}'...")
trainer.save_model(save_model_path)

# Display training results
print("Displaying training results...")
logging.info("Displaying training results...")
trainer.result()

# Evaluate on test set
print("Evaluating model on test set...")
logging.info("Evaluating model on test set...")
trainer.evaluate(test_loader, type='Test')

# Evaluate on validation set (optional)
# print("Evaluating model on validation set...")
# trainer.evaluate(val_loader, type='Validation')

# Visualize training and evaluation metrics
print("Plotting training and validation metrics...")
logging.info("Plotting training and validation metrics...")
trainer.plot_loss_accuracy()

# Generate confusion matrix
print("Generating confusion matrix...")
logging.info("Generating confusion matrix...")
conf_matrix = trainer.confusion_matrix(test_loader)
trainer.plot_confusion_matrix(conf_matrix)

# Display model architecture and save summary
print("Saving model architecture and summary...")
logging.info("Saving model architecture and summary...")
trainer.archit()

print("\nPipeline completed successfully!")
logging.info("Pipeline completed successfully!")
