# Diabetes-Prediction-Project
This project explores the use of Support Vector Machines (SVM) for predicting the presence of diabetes in individuals based on their health data. It serves as a demonstration of the machine learning process for educational purposes and should not be used for actual medical diagnosis or treatment.

## Dependencies
* pandas
* numpy
* scikit-learn

## Data
The project requires a CSV file named ```diabetes.csv``` containing relevant features and a binary target variable (```Outcome```) indicating the presence or absence of diabetes.

## Project Structure
```diabetes_prediction.py```: The main script containing the entire prediction pipeline.
Code Breakdown

### 1. Data Loading and Exploration:
* Imports necessary libraries.
* Reads the ```diabetes.csv``` data using ```pandas.read_csv```.
* Explores the data through:
 1. Displaying the first few rows (```df.head```).
 2. Checking data dimensions (```df.shape```).
 3. Analyzing descriptive statistics (```df.describe```).
 4. Examining class distribution (```df['Outcome'].value_counts```).

### 2. Data Preprocessing:
* Separates features (```x```) and target variable (```y```) using ```df.drop``` and assignment.
* Performs standard scaling using ```StandardScaler``` to ensure features are on similar scales.
* Fits the scaler on training data and transforms both training and testing data.

### 3. Model Training and Evaluation:
* Splits data into training and testing sets using ```train_test_split``` (80% for training, 20% for testing).
* Trains an SVM classifier with a linear kernel.
* Evaluates model performance using accuracy score on both training and testing sets.

### 4. Sample Prediction:
* Defines a sample data point with several health-related features.
* Reshapes and transforms the sample data for prediction.
* Makes a prediction using the trained SVM model.
* Prints a message indicating the predicted diabetic status.
