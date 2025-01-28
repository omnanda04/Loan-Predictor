# Loan Sahayak

Loan Sahayak is a web application designed to predict loan eligibility based on user inputs using machine learning models. The application predicts whether a loan will be approved or denied for an individual.

## Project Structure

The project is divided into five main parts:
1. **Data Collection**: Collects relevant data on applicants, including personal details, income, credit history, and loan status.
2. **Data Preprocessing**: Cleans and prepares the data by handling missing values, converting categorical data to numerical values, and applying feature scaling.
3. **Model Building**: Builds and trains machine learning models to predict loan eligibility, including Logistic Regression, Support Vector Classifier (SVC), and Random Forest Classifier.
4. **Backend API Development**: Develops an API to interact with the trained model and generate predictions based on user input.
5. **Frontend Development and Deployment**: A simple user interface that allows users to input their details and receive a loan eligibility prediction.

## Dataset

The dataset contains 600 rows and 13 columns, with features including:
- **Gender**
- **Marital Status**
- **Dependents**
- **Education**
- **Applicant Income**
- **Co-Applicant Income**
- **Loan Amount**
- **Credit History**
- **Loan Status** (target column: 1 for approval, 0 for denial)

## Data Preprocessing

### Steps Taken:
- **Handling Missing Values**:
  - Columns with less than 5% missing values: Rows were dropped.
  - Columns with more than 5% missing values: 
    - For numerical columns like ApplicantIncome and CoApplicantIncome, missing values were replaced with the column mean.
    - For categorical columns like SelfEmployed, missing values were replaced with the most frequent category.

- **Converting Categorical Data**: 
  - Non-numeric features like 'Married', 'Education', and 'SelfEmployed' were converted to numeric values (Yes = 1, No = 0) using Python's `map()` function.

- **Feature Scaling**:
  - Applied scaling to numerical features like ApplicantIncome, CoApplicantIncome, LoanAmount, and LoanAmountTerm to standardize their ranges.

## Model Building

### Model Selection:
- **Logistic Regression**: Great for binary classification problems.
- **Support Vector Classifier (SVC)**: Handles non-linear decision boundaries effectively.
- **Random Forest Classifier**: An ensemble method combining multiple decision trees.

### Training:
- The dataset was split into **80% training** and **20% testing** for model evaluation.
- **5-fold cross-validation** was used to evaluate the models on different subsets of data, ensuring a robust evaluation.

### Model Evaluation:
- **Logistic Regression**: 83% accuracy
- **SVC**: 82% accuracy
- **Random Forest Classifier**: 82% accuracy

### Hyperparameter Tuning:
- Focused on improving the Logistic Regression model through hyperparameter tuning.
- The accuracy of the Logistic Regression model was improved to **85%**, making it the final choice for the project.

## Testing and Saving the Model

- The final model was validated using a sample input DataFrame to ensure accurate predictions.
- The trained model was saved and integrated into the backend API for future predictions.

## Installation and Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/loan-sahayak.git
