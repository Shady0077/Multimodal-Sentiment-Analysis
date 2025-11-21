📘 Multimodal Sentiment Analysis Web Application

This repository contains a Multimodal Sentiment Analysis System that predicts sentiment using both the title and review text. The model is built using dual DistilBERT encoders and deployed using a Flask web application for easy real-time interaction.

📁 Project Structure
.
├── app.py                           # Flask web app for prediction
├── train_sentiment.py               # Script to train the multimodal model
├── multimodal_sentiment_model.pt    # Trained model weights (generated after training)
├── train.csv                        # Training dataset
├── test.csv                         # Testing dataset
├── requirements.txt                 # Project dependencies
└── templates/
    └── index.html                   # Web interface

🌟 Features

Multimodal Sentiment Analysis (Title + Review)

Dual DistilBERT Encoder Architecture

Web UI built with Flask

Supports manual text input and CSV batch prediction

Model training script included

Clean and extendable code structure

🧠 Model Architecture

The multimodal model includes:

1️⃣ DistilBERT Encoder for Title

Extracts semantic features from the title.

2️⃣ DistilBERT Encoder for Review

Extracts semantic features from the review.

3️⃣ Fusion Layer

Concatenates both [CLS] token embeddings → 1536-dimensional vector.

4️⃣ Classifier

A linear layer predicts:

0 = Negative

1 = Positive

Trained weights are saved as:

multimodal_sentiment_model.pt

🚀 Getting Started
1. Install dependencies
pip install -r requirements.txt

2. Train the Model (if weights not present)
python train_sentiment.py


This generates the file:

multimodal_sentiment_model.pt

3. Run the Flask Application
python app.py

4. Open the App in Browser
http://127.0.0.1:5000/

🖥 Web Interface Features

✔ Input title + review manually
✔ Upload CSV file for batch testing
✔ Automatically displays sentiment predictions
✔ Clean minimal HTML in /templates/index.html

📊 Dataset Format

Your CSV files follow:

label	title	review
1 or 2	product title	review text

During training:

Label 1 → Negative (0)

Label 2 → Positive (1)

📈 Model Training Workflow

Load CSV dataset

Tokenize title & review using DistilBERT tokenizer

Create PyTorch dataset and dataloaders

Train for multiple epochs

Save weights to multimodal_sentiment_model.pt

Flask app loads the model for inference

🔮 Future Enhancements

Add probability/confidence scores

Deploy on Render / Railway / AWS

Add charts for sentiment distribution

Improve UI with Bootstrap or React

Add support for neutral/multiple classes

🏁 Conclusion

This project demonstrates a complete multimodal NLP pipeline, integrating:

Dataset preparation

Dual-encoder DistilBERT architecture

Model training with PyTorch

Web deployment with Flask

A perfect project for showcasing advanced NLP, ML deployment, and full-stack integration.
