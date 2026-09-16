# Spam SMS & Email Classifier

An end-to-end machine learning project that classifies text messages as **HAM (legitimate)** or **SPAM (unwanted)** using TF-IDF feature extraction and Logistic Regression.

The project includes data preprocessing, model training, evaluation, probability-based predictions, a FastAPI backend, and a responsive web frontend.

---

## Project Overview

Spam messages can contain unwanted advertisements, suspicious links, fraudulent offers, or other unsolicited content. This project demonstrates how machine learning can help identify potential spam messages from their text.

The classifier analyzes a message and returns:

* Predicted label: HAM or SPAM
* HAM probability
* SPAM probability
* Model confidence score

**Note:** Predictions are estimates and should not be treated as a guarantee that a message is safe or malicious.

## Model Performance

The model was evaluated on a held-out test set.

| Metric         |  Score |
| -------------- | -----: |
| Accuracy       | 98.06% |
| Spam Precision | 90.30% |
| Spam Recall    | 94.53% |
| Spam F1-score  | 92.37% |

### Confusion Matrix

| Actual / Predicted | HAM | SPAM |
| ------------------ | --: | ---: |
| HAM                | 891 |   13 |
| SPAM               |   7 |  121 |

The evaluation used 1,032 test messages. The classifier correctly classified 1,012 of them.

Evaluation outputs are available in the `outputs/` directory.

## Features

* SMS spam classification
* Text preprocessing and duplicate removal
* TF-IDF feature extraction using unigrams and bigrams
* Logistic Regression classifier
* HAM and SPAM probability estimates
* Confidence score for predictions
* FastAPI prediction endpoint
* Interactive web interface
* Confusion matrix and performance visualizations
* Example predictions saved as CSV
* Automated tests for core components and API behavior

## Technology Stack

**Machine Learning**

* Python
* pandas and NumPy
* scikit-learn
* TF-IDF
* Logistic Regression
* joblib

**Backend**

* FastAPI
* Uvicorn
* Pydantic

**Frontend**

* Vite
* JavaScript
* React

**Testing and Visualization**

* pytest
* Matplotlib
* Seaborn

## Dataset

This project uses the **UCI SMS Spam Collection** dataset.

Source: https://archive.ics.uci.edu/dataset/228/sms+spam+collection

The original dataset contains 5,574 messages. After removing duplicate label-message records, 5,160 records remained.

| Label | Records after cleaning |
| ----- | ---------------------: |
| HAM   |                  4,518 |
| SPAM  |                    642 |
| Total |                  5,160 |

The dataset is used for educational and machine learning purposes. Refer to the UCI dataset page for its original description and usage information.

## Project Structure

```text
spam-sms-email-classifier/
├── backend/
│   └── main.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── samples/
├── docs/
├── frontend/
├── notebooks/
│   └── spam_classifier.ipynb
├── outputs/
│   ├── plots/
│   │   ├── confusion_matrix.png
│   │   └── metrics.png
│   ├── example_predictions.csv
│   ├── test_predictions.csv
│   ├── metrics.csv
│   └── spam_classifier.joblib
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── predict.py
├── tests/
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── vercel.json
```

## Getting Started

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd spam-sms-email-classifier
```

### 2. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Prepare the dataset

Download the UCI SMS Spam Collection dataset and place the raw `SMSSpamCollection` file at:

```text
data/raw/SMSSpamCollection
```

### 5. Train the model

```powershell
python -m src.train_model
```

The trained model is saved to:

```text
outputs/spam_classifier.joblib
```

### 6. Evaluate the model

```powershell
python -m src.evaluate_model
```

This generates the evaluation metrics, test predictions, confusion matrix, and metrics plot.

### 7. Generate example predictions

```powershell
python -m src.predict
```

Example predictions are saved to:

```text
outputs/example_predictions.csv
```

## Running the Backend API

Start the FastAPI development server from the project root:

```powershell
uvicorn backend.main:app --reload
```

The API will be available locally at:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

### API Endpoints

| Method | Endpoint   | Purpose                          |
| ------ | ---------- | -------------------------------- |
| GET    | `/`        | API information and model status |
| GET    | `/health`  | Health check                     |
| POST   | `/predict` | Classify a message               |

### Example Prediction Request

```json
{
  "message": "Congratulations! You have won a free prize. Call now!"
}
```

### Example Response

```json
{
  "predicted_label": "spam",
  "confidence": 0.9842,
  "ham_probability": 0.0158,
  "spam_probability": 0.9842
}
```

The exact probability values can vary if the model is retrained or changed.

## Running the Frontend

Open a separate terminal:

```powershell
cd frontend
npm install
```

Configure the frontend API URL in `frontend/.env`:

```env
VITE_API_URL=http://127.0.0.1:8000
```

Start the frontend:

```powershell
npm run dev
```

Open the local URL displayed in the terminal, usually:

```text
http://localhost:5173
```

Ensure the backend is running before submitting messages.

## Testing

Run the complete test suite from the project root:

```powershell
python -m pytest
```

The project test suite currently contains **58 passing tests**.

## Outputs

The `outputs/` directory contains:

* `spam_classifier.joblib` — serialized machine learning pipeline
* `metrics.csv` — evaluation metric values
* `test_predictions.csv` — test-set predictions and probability estimates
* `example_predictions.csv` — example messages with predicted labels and confidence scores
* `plots/confusion_matrix.png` — confusion matrix visualization
* `plots/metrics.png` — performance metrics visualization

## Deployment

The project is configured for Vercel deployment.

The deployment includes a FastAPI backend and a frontend that communicates with the backend through the `VITE_API_URL` environment variable.

**Live application:** To be added after successful deployment.

**Backend API:** To be added after successful deployment.

Deployment URLs will be added after the live application and API have been tested.

## Limitations

* The model was trained on the UCI SMS Spam Collection dataset; performance on other message types or real-world data may differ.
* The model uses message text and does not verify URLs, sender identities, or external information.
* Probability and confidence values are model estimates, not calibrated guarantees.
* A message classified as HAM may still be harmful or fraudulent.

## Future Improvements

* Evaluate on additional SMS and email datasets.
* Improve handling of unseen words and unusual message formats.
* Add more extensive error analysis.
* Explore additional classification algorithms.
* Improve deployment monitoring and API security.

## Acknowledgements

Dataset: UCI Machine Learning Repository — SMS Spam Collection.

https://archive.ics.uci.edu/dataset/228/sms+spam+collection

## License

See the `LICENSE` file in this repository.
<br>
WEBSITE URL:
<br>
https://spam-sms-email-classifier-4z9j.vercel.app/
