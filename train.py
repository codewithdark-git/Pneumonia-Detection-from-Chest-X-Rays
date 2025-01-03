import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchviz import make_dot
from torchsummary import summary
import itertools
from sklearn.metrics import confusion_matrix
import tqdm
import logging

class Train:
    
    def __init__(self, model, device, train_loader, batch_size, num_epochs=2, learning_rate=0.01, model_name=None):
        self.model = model
        self.model_name = model_name or 'Untitled'
        self.device = device
        self.train_loader = train_loader
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size= batch_size

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.train_losses = []  # To store training losses
        self.train_accuracies = []  # To store training accuracies
        self.validation_accuracies = []  # To store validation accuracies


        self.logging = logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(levelname)s - %(message)s",
                handlers=[
                    logging.FileHandler(f"{self.model_name}.log"),
                    logging.StreamHandler(),
                ],
            )


    def trainer(self):
        """
        Train the model using Adam optimizer and CrossEntropyLoss.

        Returns:
        None
        """

        print(f"""
              
                ****************************************
                ||=    Training PneumoniaClassifier
                ||=    -------------------------------
                ||=    Batch size: {self.batch_size}
                ||=    Learning rate: {self.learning_rate}
                ||=    Epochs: {self.num_epochs}
                ||=    Device: {self.device}
                ||=    Model : {self.model_name}
                ****************************************
        
        """)


        for epoch in range(self.num_epochs):
            self.model.train()
            progress_bar = tqdm.tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs}", leave=True)
            running_loss = 0.0
            total_preds = 0
            correct_preds = 0

            for i, (inputs, labels) in enumerate(progress_bar):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()

                # Track loss and accuracy
                _, predicted = torch.max(outputs, 1)
                total_preds += labels.size(0)
                correct_preds += (predicted == labels).sum().item()
                running_loss += loss.item()

                progress_bar.set_description(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {loss.item():.4f}")

            epoch_loss = running_loss / len(self.train_loader)
            epoch_accuracy = 100 * correct_preds / total_preds

            self.train_losses.append(epoch_loss)
            self.train_accuracies.append(epoch_accuracy)

            print(f"Epoch {epoch + 1}/{self.num_epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
            self.logging.info(f"Epoch {epoch + 1}/{self.num_epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

    def evaluate(self, loader, type='Validation'):
        """
        Evaluate the model on a validation or test dataset.

        Returns:
        None
        """
        self.model.eval()
        total_preds = 0
        correct_preds = 0
        running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_preds += labels.size(0)
                correct_preds += (predicted == labels).sum().item()

        validation_accuracy = 100 * correct_preds / total_preds
        self.validation_accuracies.append(validation_accuracy)

        print(f"{type} Loss: {running_loss / len(loader):.4f}, {type} Accuracy: {validation_accuracy:.2f}%")
        self.logging.info(f"{type} Loss: {running_loss / len(loader):.4f}, {type} Accuracy: {validation_accuracy:.2f}%")

    def result(self):
        """
        Print final training loss and accuracy.
        """
        print("\n" + "=" * 50)
        print(f"TRAINING RESULTS")
        print("=" * 50)
        print(f"Number of Epochs: {self.num_epochs}")
        print("-" * 50)
        print(f"Final Training Loss: {self.epoch_losses[-1]:.4f}")
        print(f"Final Training Accuracy: {self.epoch_accuracies[-1]:.2f}%")
        print("=" * 50)
        self.logging.info("Final Training Accuracy: {self.epoch_accuracies[-1]:.2f}%")
        self.logging.info("Final Training Loss: {self.epoch_losses[-1]:.4f}")



    def plot_loss_accuracy(self):
        """
        Visualize training loss and accuracy over epochs.
        """
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

        # Plot training loss
        ax1.plot(range(1, self.num_epochs + 1), self.epoch_losses, 'b-', label='Training Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss Over Time')
        ax1.grid(True)
        ax1.legend()

        # Plot training accuracy
        ax2.plot(range(1, self.num_epochs + 1), self.epoch_accuracies, 'g-', label='Training Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training Accuracy Over Time')
        ax2.grid(True)
        ax2.legend()

        plt.tight_layout()
        plt.savefig('plot_loss_accuracy.png')
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
        plt.savefig('training_validation_accuracy.png')
        plt.show()

    def save_model(self, path):
        """
        Save the trained model to a specified path.
        """
        torch.save(self.model.state_dict(), path)
        print(f"Model saved successfully at {path}")

    def archit(self):
        """
        Save model summary and architecture visualization.
        """
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        summary(self.model, (3, 224, 224))
        print("Model summary saved.")
        self.logging.info("Model summary saved.")
        self.logging.info(f'su')
        dot = make_dot(self.model(dummy_input), params=dict(self.model.named_parameters()))
        dot.render("model_architecture", format="png")
        print("Model architecture saved as 'model_architecture.png'.")
        self.logging.info("Model architecture saved as 'model_architecture.png'.")

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
        plt.savefig('confusion_matrix.png')
        self.logging.info("Confusion Matrix saved as 'confusion_matrix.png'.")
