# Multimodal Sentiment Analysis

## 🎯 Project Overview
This project implements a sophisticated **multi-modal sentiment analysis system**. Its primary function is to predict whether a user review is **Positive** or **Negative** by simultaneously analyzing two distinct text inputs: **the review title** and **the full review text**.

The system leverages the power of deep learning, combining two separate DistilBERT Transformer models into a custom-built neural network for enhanced prediction accuracy, and is deployed as a user-friendly web application using Flask.

## ✨ Features

* **Multimodal Fusion:** Analyzes both the review title and the main text for a more accurate sentiment prediction.
* **Dual Transformer Architecture:** Utilizes two separate DistilBERT models (one for each input modality) for powerful feature extraction.
* **Custom Neural Network Head:** A custom-built neural network layer combines the outputs of the two transformers for the final classification.
* **Web Deployment:** Deployed via a lightweight [Flask](https://flask.palletsprojects.com/) web application with an intuitive user interface (`templates/index.html`).
* **Python/Machine Learning Focus:** Core logic is implemented in Python, leveraging popular ML/NLP libraries.

## 🛠️ Technology Stack

| Category | Technology | Notes |
| :--- | :--- | :--- |
| **Language** | Python | Primary development language. |
| **NLP Models** | DistilBERT | Used for high-performance text encoding. |
| **Deep Learning** | (Likely PyTorch or TensorFlow) | Framework used for building and training the custom model head. |
| **Web Framework** | Flask | Used for serving the model as a web application. |
| **Dependencies** | Listed in `requirements.txt`. | e.g., `pandas`, `scikit-learn`, `transformers`. |
| **Frontend** | HTML/CSS | Used for the web application interface. |

## 🚀 Installation and Setup

Follow these steps to set up the project locally.

### 1. Clone the Repository

```bash
git clone [https://github.com/Shady0077/Multimodal-Sentiment-Analysis.git](https://github.com/Shady0077/Multimodal-Sentiment-Analysis.git)
cd Multimodal-Sentiment-Analysis