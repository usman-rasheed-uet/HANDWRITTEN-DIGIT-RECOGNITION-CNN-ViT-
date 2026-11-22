===========================================================
HANDWRITTEN DIGIT RECOGNITION (CNN + ViT)
Student: Usman Rasheed
Roll No: TRN 2026 000313
===========================================================

INSTALL DEPENDENCIES (one time):
   pip install torch torchvision scikit-learn streamlit pillow numpy matplotlib

FOLDER CONTENTS:
   - app.py                  Streamlit app
   - cnn_digit.pt            Trained CNN weights
   - vit_digit.pt            Trained ViT weights
   - preprocess_meta.json    Preprocessing settings
   - Digit_Project.ipynb     Training notebook
   - evaluation_report.md    Test accuracy, F1, confusion matrices
   - test_images/            200 sample handwritten digits
   - sample_for_app.png      Demo image

HOW TO RUN THE APP:
   1. Open terminal in this folder
   2. Run:  streamlit run app.py
   3. Open http://localhost:8501 in browser
   4. Upload any PNG/JPEG containing one handwritten digit
   5. See CNN and ViT predictions side by side

HOW TO RETRAIN (optional):
   1. Open Digit_Project.ipynb in Google Colab
   2. Run all cells top to bottom
   3. Downloads cnn_digit.pt, vit_digit.pt, preprocess_meta.json at the end

ERROR HANDLING TESTS:
   - Blank image  → rejected with "Blank image (no strokes)"
   - Broken file  → rejected with "could not be opened as an image"
   - No upload    → shows "Waiting for an image"

NOTES:
   - Training data: sklearn.datasets.load_digits() only
   - Both models trained on identical stratified 60/20/20 split
   - Real MNIST test images used only for demo (not training)
   - Model input: 8x8 grayscale, values 0-1
   - Prepared 8x8 preview shown in the app before prediction
===========================================================