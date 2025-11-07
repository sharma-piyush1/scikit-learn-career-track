# Scikit-Learn Career Track — End-to-End Machine Learning Pipeline

This repository contains a complete, production-ready machine learning pipeline built using scikit-learn.  
It demonstrates how to go from raw data → preprocessing → modeling → evaluation → interpretability → deployment API.

## 📊 Dataset
Titanic survival dataset  
Goal: Predict passenger survival (classification)

## 🧱 Project Structure

notebooks/ # Complete learning & experimentation workflow
model/ # Saved production-ready model pipeline
app/ # FastAPI service for model inference
requirements.txt # Reproducible environment
README.md # Documentation


## ✅ Key Skills Demonstrated

| Skill | Description |
|------|-------------|
| Data Cleaning & Preprocessing | Handling missing values, encoding, scaling |
| Feature Engineering | Interaction terms, polynomial features |
| Pipelines & ColumnTransformer | **Reusable** and **production-safe** workflows |
| Model Training & Evaluation | Classification + regression models |
| Cross-Validation | Stable model comparison |
| Hyperparameter Tuning | RandomizedSearchCV optimization |
| Model Interpretability | SHAP feature importance analysis |
| Deployment | FastAPI inference endpoint with Pydantic validation |

## 🧪 Run FastAPI Locally

```bash
uvicorn app.app:app --reload --port 8000

http://127.0.0.1:8000/docs

### 🎯 Outcome

This project simulates a real ML workflow suitable for:

ML Engineer Job Preparation

Portfolio Showcase

End-to-End Data Science Skill Demonstration
