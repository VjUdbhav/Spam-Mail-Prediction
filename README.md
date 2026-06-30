# Spam Mail Prediction

A beginner-friendly machine learning project that classifies email/SMS messages as **Spam** or **Ham** (not spam) using **TF-IDF** text feature extraction and a **Logistic Regression** classifier.

## Project Overview

This project builds a simple text classification pipeline to detect spam messages. It walks through loading raw text data, cleaning and encoding labels, converting text into numerical features using TF-IDF, training a Logistic Regression model, and evaluating its performance. A reusable prediction function is also included so you can test the model on your own custom messages.

## Folder Structure

```
Spam_Mail_Prediction/
├── Spam_Mail_Prediction.ipynb   # Main Jupyter notebook with full workflow
├── README.md                    # Project documentation
├── dataset/
│   └── spam.csv                 # Raw dataset (message + category)
└── output/
    └── class_distribution.png   # Bar chart of spam vs ham counts
```

## Dataset

- **File:** `dataset/spam.csv`
- **Rows:** 5,572 messages
- **Columns:**
  - `Category` — label, either `ham` (not spam) or `spam`
  - `Message` — the raw text content of the email/SMS
- **Class distribution:** ~4,825 ham messages and ~747 spam messages (imbalanced, as is typical for spam datasets)
- No missing values present in the dataset.

## Libraries Used

- `numpy` — numerical operations
- `pandas` — data loading and manipulation
- `matplotlib` — visualizing class distribution
- `scikit-learn`
  - `train_test_split` — splitting data into train/test sets
  - `LabelEncoder` — encoding `ham`/`spam` labels into 0/1
  - `TfidfVectorizer` — converting text into numerical TF-IDF features
  - `LogisticRegression` — classification model
  - `accuracy_score` — model evaluation metric

## Setup

1. Clone or download this project folder.
2. Install the required libraries:
   ```bash
   pip install numpy pandas matplotlib scikit-learn jupyter
   ```
3. Make sure `spam.csv` is present inside the `dataset/` folder.

## Usage

1. Open the notebook:
   ```bash
   jupyter notebook Spam_Mail_Prediction.ipynb
   ```
2. Run all cells in order, from top to bottom.
3. The notebook will:
   - Load and explore the dataset
   - Encode labels and split data
   - Extract TF-IDF features
   - Train a Logistic Regression model
   - Print training and testing accuracy
   - Save a class distribution plot and accuracy results to `output/`
4. Use the final cell to test your own message:
   ```python
   sample_message = "Your custom message here"
   result = predict_message(sample_message)
   print(result)  # Outputs "Spam" or "Ham"
   ```

## Sample Output

```
Training Accuracy: 96.7%
Testing Accuracy:  96.6%

Message: Congratulations! You've won a free iPhone. Click here to claim your prize now!
Prediction: Spam

Message: Hey, are we still meeting for lunch tomorrow?
Prediction: Ham
```

## Notes

- The model is intentionally kept simple (TF-IDF + Logistic Regression) for clarity and beginner-friendliness, prioritizing readability over squeezing out maximum accuracy.
- The TF-IDF vectorizer is fit only on training data and used to transform both train and test sets, avoiding data leakage.
- Class imbalance is present but not explicitly handled (e.g., no SMOTE), keeping the project simple — accuracy is still strong as a result of TF-IDF being highly effective for this kind of text data.
