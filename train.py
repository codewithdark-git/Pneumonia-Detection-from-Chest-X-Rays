import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchsummary import summary
import numpy as np
import tqdm
import logging
import os
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import makedot



class PneumoniaTrainer:
    """
    A class for training a CNN model for pneumonia detection.
    """
    def __init__(self, model, model_name, img_size=50, batch_size=100, learning_rate=0.001, epochs=2, model_path="Model/model.pth"):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = img_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.model_path = model_path
        self.model_name = model_name

        # Initialize model, optimizer, and loss function
        if model is None:
            raise ValueError("Please provide a valid model instance.")
        self.net = model.to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.loss_function = nn.CrossEntropyLoss()


    def train_model(self, train_loader):
        """Train the model with progress tracking using tqdm."""
        print(f"""
              
                ****************************************
                ||=    Training MelanomaClassifier
                ||=    -------------------------------
                ||=    Batch size: {self.batch_size}
                ||=    Learning rate: {self.learning_rate}
                ||=    Epochs: {self.epochs}
                ||=    Model path: {self.model_path}
                ||=    Device: {self.device}
                ||=    Model : {self.model_name}
                ****************************************
        
            """)
        self.net.to(self.device)
        self.net.train()  # Set model to training mode

        for epoch in range(self.epochs):
            progress_bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}", leave=True)
            running_loss = 0.0

            for i, (inputs, labels) in enumerate(progress_bar):
                # Move data to the same device as the model
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.net(inputs)
                loss = self.loss_function(outputs, labels)

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()

                # Update running loss
                running_loss += loss.item()

                # Update progress bar description
                progress_bar.set_description(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss.item():.4f}")

            # Log epoch loss
            epoch_loss = running_loss / len(train_loader)
            logging.info(f"Epoch {epoch + 1}/{self.epochs} Loss: {epoch_loss:.4f}")

    def evaluate_accuracy(self, data_loader, dataset_name="Validation"):
        """Evaluates the accuracy on the given dataset."""
        self.net.eval()  # Set model to evaluation mode
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.net(inputs)
                predicted = torch.argmax(outputs, dim=1)

                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        accuracy = (correct / total) * 100
        logging.info(f"{dataset_name} Accuracy: {accuracy:.2f}%")
        print(f"{dataset_name} Accuracy: {accuracy:.2f}%")

    def summary_model(self):
        """Displays a summary of the model architecture."""
        return summary(self.net, input_size=(1, 224, 224), device=str(self.device))


    def plot_loss_accuracy(self):
        """
        Visualize training loss and accuracy over epochs.
        """
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

        # Plot training loss
        ax1.plot(range(1, self.num_epochs + 1), self.train_losses, 'b-', label='Training Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss Over Time')
        ax1.grid(True)
        ax1.legend()

        # Plot training accuracy
        ax2.plot(range(1, self.num_epochs + 1), self.train_accuracies, 'g-', label='Training Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training Accuracy Over Time')
        ax2.grid(True)
        ax2.legend()

        plt.tight_layout()
        plt.savefig('Images/plot_loss_accuracy.png')
        plt.show()

    def plot_train_test_accuracy(self):
        """
        Plot both training and validation accuracy over epochs.
        """
        plt.figure(figsize=(10, 6))

        # Plot training accuracy
        plt.plot(range(1, self.num_epochs + 1), self.train_accuracies, 'g-', label='Training Accuracy')

        # Plot validation accuracy if available
        if len(self.validation_accuracies) == self.num_epochs:
            plt.plot(range(1, self.num_epochs + 1), self.validation_accuracies, 'b-', label='Validation Accuracy')

        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.title('Training and Validation Accuracy Over Epochs')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save the plot
        plt.savefig('Images/training_validation_accuracy.png')
        plt.show()

    def save_model(self, path):
        """
        Save the trained model to a specified path.
        """
        torch.save(self.model, path)
        print(f"Model saved successfully at {path}")

    def archit(self):
        """
        Save model summary and architecture visualization.
        """
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        summary(self.model, (3, 224, 224))
        print("Model summary saved.")
        self.logger.info("Model summary saved.")
        self.logger.info(f'su')
        dot = make_dot(self.model(dummy_input), params=dict(self.model.named_parameters()))
        dot.render("Images/model_architecture", format="png")
        print("Model architecture saved as 'model_architecture.png'.")
        self.logger.info("Model architecture saved as 'model_architecture.png'.")

    def plot_confusion_matrix(self, loader, classes, cmap=plt.cm.Blues):
        """
        Plot confusion matrix.

        Returns:
        None
        """
        test_labels = []
        predicted_labels = []

        self.model.eval()
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                test_labels.extend(labels.cpu().numpy())
                predicted_labels.extend(preds.cpu().numpy())

        cm = confusion_matrix(test_labels, predicted_labels)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.show()
        plt.savefig('Images/confusion_matrix.png')
        self.logger.info("Confusion Matrix saved as 'confusion_matrix.png'.")
