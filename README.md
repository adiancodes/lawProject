# Pocket Legal Assistant

An academic machine learning project that classifies everyday legal problems described in plain language and returns the relevant statute and a suggested course of action. The project is built entirely in Python and deployed as a web application using Streamlit.

Live at: https://meetyourpersonallawbuddy.streamlit.app

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
├── app.py                # Streamlit web application
├── train_model.py        # Model training script (TF-IDF + SVC with GridSearchCV)
├── legal_model.pkl       # Trained SVC classifier
├── vectorizer.pkl        # Fitted TF-IDF vectorizer
├── requirements.txt      # Pinned dependencies for Streamlit Cloud
└── README.md
```

The `.pkl` files are committed directly to the repository. Streamlit Cloud serves the app without executing the training script, so the pre-trained model artifacts must be present in the repo.

---

## How It Works

**Training (`train_model.py`):**
The training script loads the labelled dataset, splits it 80/20 into training and test sets, and vectorises the text using a TF-IDF vectorizer configured with unigram and bigram features. It then runs a grid search over eight combinations of the SVC regularisation parameter C and kernel type using 5-fold cross-validation to find the best performing configuration. The best model and vectorizer are saved to disk as `.pkl` files.

**Application (`app.py`):**
The Streamlit application loads the saved model and vectorizer. The user types a description of their legal situation into a text field and clicks Analyse. The input is transformed using the loaded vectorizer and passed to the model, which predicts the legal category. The application then displays the classification along with the relevant law and a recommended next step.

---

## Running Locally

**Prerequisites:** Python 3.8 or later.

Install the required packages:

```
pip install -r requirements.txt
```

Launch the application:

```
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

To retrain the model (for example after updating the dataset), run:

```
python train_model.py
```

This will overwrite `legal_model.pkl` and `vectorizer.pkl` with the newly trained artifacts.

---

## Deployment on Streamlit Cloud

The app is deployed via Streamlit Community Cloud. The dependencies are pinned in `requirements.txt` to match the scikit-learn version used to train the model, which prevents version mismatch errors when loading the `.pkl` files.

If you fork this repository and deploy your own instance, ensure that `legal_model.pkl` and `vectorizer.pkl` are committed to the repo before connecting it to Streamlit Cloud.

---

## Model Performance

The dataset contains approximately 195 labelled examples across nine categories, which works out to roughly 20 samples per class. Given this constraint, the model achieves around 55 percent accuracy on the held-out test set, which is reasonable for the dataset size. The best configuration found by grid search was a linear kernel SVM with C set to 10.

Accuracy will improve significantly with a larger and more varied dataset.

---

## Limitations

This is an academic project and the model has real limitations that should be understood before use.

- The dataset is small. With only around 20 examples per category, the model can misclassify scenarios that use unusual phrasing or describe edge cases not represented in the training data.
- The legal responses are static and pre-written. They are general in nature and do not account for the specifics of an individual case, applicable state laws, or recent amendments.
- The model was trained on Indian legal contexts. The statutes and next steps referenced are specific to the Indian legal system.

**This tool does not provide professional legal advice. It is intended for academic demonstration purposes only. For any real legal matter, please consult a qualified advocate.**

---

## Acknowledgements

The web application is built with Streamlit. The classification model uses scikit-learn's Support Vector Classifier.
