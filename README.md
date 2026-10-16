#  Loan Eligibility Prediction System

##  Project Overview

The Loan Eligibility Prediction System is a Machine Learning project that predicts whether a loan applicant is eligible or not eligible based on their financial information.

This project uses Logistic Regression and Decision Tree classification algorithms to analyze applicant details and predict loan eligibility. It also includes data visualization to compare model accuracy and understand relationships between financial attributes.

##  Objectives

- Predict loan eligibility based on applicant financial details.
- Analyze income, credit history, loan amount, and existing debt.
- Compare Logistic Regression and Decision Tree algorithms.
- Visualize data using different types of graphs.
- Understand basic Machine Learning classification techniques.

##  Features

- User input for applicant financial details.
- Loan eligibility prediction using two ML algorithms.
- Model accuracy comparison using a bar graph.
- Income and loan amount visualization using a line graph.
- Income and debt relationship analysis using a scatter plot.
- Eligible and not-eligible distribution using a pie chart.
- Model evaluation using a confusion matrix.

## 🛠️ Technologies Used

- Programming Language: Python
- Data Analysis: Pandas, NumPy
- Machine Learning: Scikit-learn
- Data Visualization: Matplotlib, Seaborn
- Development Environment: VS Code

##  Input Features

 Feature   -   Description 

 Income    - Applicant's income 
 Credit    - Credit history (1 = Good, 0 = Poor) 
 Loan      - Requested loan amount 
 Debt      - Existing debt obligations 
 Status    - Target label (1 = Eligible, 0 = Not Eligible) 

##  Machine Learning Algorithms

### 1. Logistic Regression
A classification algorithm used to predict loan eligibility based on applicant details.

### 2. Decision Tree Classifier
A tree-based classification algorithm that predicts loan eligibility using learned decision rules.

Both models are evaluated using accuracy on a test dataset.

##  Data Visualizations

The project includes the following visualizations:

1. Bar Graph: Compares the accuracy of the two ML models.
2. Line Graph: Displays income versus loan amount.
3. Scatter Plot: Shows the relationship between income and existing debt.
4. Pie Chart: Displays the eligible and not-eligible distribution in the dataset.
5. Confusion Matrix: Compares actual and predicted classifications.

##  Project Structure

text
loan-eligibility-prediction/
│
├── loan_prediction.py
├── requirements.txt
└── README.md

##  Installation and Setup

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual GitHub repository URL.

### Step 2: Open the Project Folder

```bash
cd loan-eligibility-prediction
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Python Program

```bash
python loan_prediction.py
```

##  Sample Input

text
Enter income: 50000
Credit history (1=Good, 0=Poor): 1
Enter loan amount: 200000
Enter existing debt: 5000


##  Output

The program displays:

- Logistic Regression accuracy.
- Decision Tree accuracy.
- Loan eligibility prediction from both models.
- Model accuracy comparison graph.
- Income versus loan amount line graph.
- Income versus debt scatter plot.
- Loan eligibility pie chart.
- Logistic Regression confusion matrix.

Note: Actual accuracy and prediction results depend on the dataset and trained models.

##  Dataset Information

This project uses a small sample dataset created directly in Python for demonstration and learning purposes. It is not a real banking dataset.

##  Limitations

- The dataset contains only 15 sample records.
- The model's predictions are not suitable for real banking or financial decisions.
- Accuracy on a small dataset may not represent real-world performance.
- A larger, representative dataset and proper validation are required for reliable predictions.

##  Future Enhancements

- Integrate a real-world loan eligibility dataset.
- Improve model performance through feature engineering and hyperparameter tuning.
- Add precision, recall, F1-score, and ROC-AUC evaluation.
- Save and load trained models using Joblib.
- Develop a web interface using Flask or Streamlit.
- Add more applicant features, such as employment status and loan tenure.


