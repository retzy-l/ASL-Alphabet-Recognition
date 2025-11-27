Project: Real-Time ASL Alphabet Recognition
Course: CENG 476 - Deep Learning
Group Members: ABDIRAHMAN AHMED HUSSEIN (230446614) 
               İLTER KARAMÜFTÜOĞLU (200444078)

📁 Project Structure

ASL_Project/
│
├── main.py               # Training script (CNN + augmentation + scheduler)
├── live_demo.py          # Real-time ASL recognition interface
├── create_charts.py      # Generates confusion matrix & accuracy charts
├── asl_robot_brain.pth   # Saved model weights
├── asl_alphabet_train/   # Dataset folder (29 classes)
├── error_gallery/        # Auto-generated misclassified images
├── loss_curve.png
├── accuracy_curve.png
├── confusion_matrix.png
└── class_accuracy_chart.png


=== HOW TO RUN ===

1. PREREQUISITES
   Install the required libraries using pip:
   pip install torch torchvision opencv-python pyttsx3 scikit-learn seaborn matplotlib

2. Model Architecture (Custom CNN)
Input: 64×64 RGB Image
Feature Extractor:
- Conv(3→32) → BN → ReLU → MaxPool
- Conv(32→64) → BN → ReLU → MaxPool
- Conv(64→128) → BN → ReLU → MaxPool

Classifier:
- Flatten
- Dropout(0.5)
- Linear(8192 → 512)
- ReLU
- Dropout(0.5)
- Linear(512 → 29)

3. TRAINING (Optional)
   The model is already trained and saved as 'asl_robot_brain.pth'.
   To retrain from scratch (25 Epochs):
   > python main.py
   (Note: This script enforces CPU usage to avoid compatibility issues with newer GPUs).

4. REPORT CHARTS
   To generate the Confusion Matrix and Accuracy/Loss plots:
   > python create_charts.py
   (Output files: confusion_matrix.png, class_accuracy_chart.png)

5. LIVE DEMO (The Main Application)
   To run the real-time assistive interface:
   > python live_demo.py

   Controls:
   - Hold a sign for 1.5s: Auto-types the letter.
   - ENTER: Speaks the current sentence aloud.
   - SPACE: Adds a space.
   - D: Deletes the last character.
   - Q: Quits the application.
   - A-Z Keys: 'Teacher Mode' (Saves the current frame as a correction for that letter).

6. NOTES
   - Ensure the 'asl_alphabet_train' folder is in the root directory.
   - 'asl_robot_brain.pth' must be present for the demo to work.