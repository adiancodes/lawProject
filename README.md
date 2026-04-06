# Pocket Legal Assistant

An academic machine learning project that classifies everyday legal problems described in plain language and returns the relevant statute and a suggested course of action. The project is built entirely in Python and runs as a local web application using Streamlit.

---

## Overview

People often encounter legal problems — a landlord who refuses to return a deposit, a phone snatched on the street, an employer withholding salary — but have no easy way to understand which area of law applies or where to begin. This project attempts to bridge that gap in a rudimentary way.

The system takes a user's description of their situation, runs it through a trained text classification model, and returns the legal category it falls under along with a plain-language explanation of the relevant law and a practical first step.

The nine categories the model can identify are:

- Traffic Harassment
- Tenant Rights
- Cybercrime (Financial Fraud)
- Cybercrime (Harassment)
- Consumer Protection
- Employment Dispute
- Property Dispute
- Public Nuisance
- Theft

---

## Project Structure

```
Law/
├── expanded_legal_dataset.csv   # Training data: ~195 labelled scenarios
├── generate_data.py             # Script used to generate training data via Gemini API
├── train_model.py               # Model training script (TF-IDF + SVC with GridSearchCV)
├── app.py                       # Streamlit web application
├── legal_model.pkl              # Saved trained model (generated after training)
├── vectorizer.pkl               # Saved TF-IDF vectorizer (generated after training)
└── README.md
```

---

## How It Works

**Training (`train_model.py`):**
The training script loads the CSV dataset, splits it 80/20 into training and test sets, and vectorises the text using a TF-IDF vectorizer configured with unigram and bigram features. It then runs a grid search over eight combinations of the SVC regularisation parameter C and kernel type using 5-fold cross-validation to find the best performing configuration. The best model and vectorizer are saved to disk as `.pkl` files.

**Application (`app.py`):**
The Streamlit application loads the saved model and vectorizer. The user types a description of their legal situation into a text field and clicks Analyse. The input is transformed using the loaded vectorizer and passed to the model, which predicts the legal category. The application then displays the classification and a pre-written response containing the relevant law and a recommended next step.

---

## Setup and Installation

**Prerequisites:** Python 3.8 or later.

Install the required packages:

```
pip install scikit-learn pandas joblib streamlit
```

**Step 1 — Train the model:**

Run the training script once to generate the model files. This will print the best hyperparameters, accuracy score, and a full classification report to the console.

```
python train_model.py
```

**Step 2 — Launch the application:**

```
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

---

## Model Performance

The dataset contains approximately 195 labelled examples across nine categories, which works out to roughly 20 samples per class. Given this constraint, the model achieves around 55 percent accuracy on the held-out test set, which is reasonable for the dataset size. The best configuration found by grid search was a linear kernel SVM with C set to 10.

Accuracy will improve significantly with a larger and more varied dataset. The `generate_data.py` script can be used to expand the training data further using the Gemini API.

---

## Limitations

This is an academic project and the model has real limitations that should be understood before use.

- The dataset is small. With only around 20 examples per category, the model can misclassify scenarios that use unusual phrasing or describe edge cases not present in the training data.
- The legal responses are static and pre-written. They are general in nature and do not account for the specifics of an individual case, the applicable state laws, or recent amendments.
- The model was trained on Indian legal contexts. The statutes and next steps referenced are specific to the Indian legal system.

**This tool does not provide professional legal advice. It is intended for academic demonstration purposes only. For any real legal matter, please consult a qualified advocate.**

---

## Acknowledgements

Training data was generated using the Gemini API from Google. The web application is built with Streamlit. The classification model uses scikit-learn's Support Vector Classifier.
