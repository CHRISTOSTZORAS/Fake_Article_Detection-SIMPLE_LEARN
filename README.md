# 📰 Fake Article Detection 🧠
## 📖 Project Overview
This project is designed to classify news articles into fake and true categories using machine learning techniques. The goal is to build a model that can predict whether a news article is fake or true.

### Data Preparation

- **Loading Data**: The code loads two CSV files, one containing fake news articles (`Fake.csv`) and one containing true news articles (`True.csv`).

- **Labeling the Data**:
  - It assigns `0` (Fake News) to the `class` column for the fake news dataset.
  - It assigns `1` (True News) to the `class` column for the true news dataset.

### Manual Testing Data

- **Selecting Last 10 Rows**: The last 10 rows from each dataset are selected for manual testing (fake and true news).
  
- **Dropping Rows**: The last 10 rows of both datasets are removed from the main dataset after selection for manual testing.

### Data Cleaning

- **Removing Unnecessary Columns**: The columns `title`, `subject`, and `date` are dropped because they are not needed for this task.

- **Text Preprocessing**:
  - The `wordopt` function is applied to clean the text by removing URLs, special characters, digits, and other unwanted elements.
  - The text is converted to lowercase and cleaned for further processing.

### Feature Extraction and Data Splitting

- **TF-IDF Vectorization**: The `TfidfVectorizer` is used to convert the cleaned text into numerical features (vector representation).

- **Train-Test Split**: The dataset is split into training and testing sets, with 75% used for training and 25% for testing.

### Model Training

- **Logistic Regression**:
  - A logistic regression model is trained on the training data.
  - The model is evaluated on the test set, and predictions are made.
  - The classification report (precision, recall, f1-score) is printed for the logistic regression model.

- **Decision Tree Classifier**:
  - A decision tree model is also trained on the same data.
  - Predictions are made using the trained decision tree model.
  - The classification report for the decision tree model is printed as well.

### Manual Testing Function

The `manual_testing` function allows you to manually input a news article and predict whether it is fake or true using the trained models.

- The text is preprocessed and then transformed into the same format used for training.
- Predictions are made using both the logistic regression and decision tree models.

### User Input for Manual Testing

- **Input Prompt**: The user is prompted to enter the news text they want to test.
  
- **Prediction Output**: The manual test results are printed, showing the prediction from both the Logistic Regression (LR) and Decision Tree (DT) models.
## 🛠️ Tech Stack & Tools

- **Programming Language:** Python 
- **Machine Learning:** Logistic Regression, Decision Tree  


## 🖥️ Run the Project

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/fake-article-detection.git
   cd fake-article-detection
   ``` 
2. **Create a Virtual Environment :**
   ```bash
   python -m venv myvenv
   ```
3. **Activate It**
   ```bash
    .\myvenv\Scripts\activate
   ```
4. **Install Dependencies**
   ```
   pip install -r .\requirements.txt
   ```

## 🤝 Contributing 

Contributions are welcome! To contribute:

* Fork the repository.
* Create a new branch: git checkout -b feature-branch
* Commit your changes: git commit -m 'Add new feature'
* Push to the branch: git push origin feature-branch
* Submit a Pull Request