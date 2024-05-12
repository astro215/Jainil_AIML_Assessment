# A.P. Moller Maersk Case Study - DS_ML Coding Challenge

This repository contains a Jupyter notebook that aims to solve the DS_ML Coding Challenge provided by a company. The challenge involves using machine learning models to predict the sourcing cost and quantity for June 2021 based on a given dataset.

Jupyter Notebook Rendered HTML with interactive plots - https://astro215.github.io/test-n/test.html
Juptyer Notebook Code - 

## Problem Statement

The problem statement is as follows:

1. Dataset: Every row represents the sourcing of one unit of a particular product combination. A unique product combination comprises of attributes mentioned in Columns A,B,C,D,E,F. Since each row represents 1 unit of sourcing; therefore, you will find multiple rows with the same combination in the training dataset. Imagine buying 1 quantity being represented as a single row. July 20 to May 21 is your training set and June 21 is your test set; So using the 11 months data (Training Set: June 2020 to May 2021) you'd have the forecast / predict the June 2021 number (Test Set) June 2021 has only a single value for each combination as that is your test set (target).
2. Problem: Iterate on ML models to come up closest to the Test set data using the Training Set Data.
3. What is expected: Understand the data set (even with the open questions you have) Do Exploratory Data Analysis. Use Python and it's libraries for all your development needs. Have a strategy for handling outliers / poor data quality on some rows. Come up with approaches for forecasting the June 21 test set. Compare and explain the different approaches you might have considered. (In your notebook) Explain the final approach you have taken and why. (In your notebook)
4. What you need to share: Share your (Jyupiter) notebook. Share your Github Repo with final solution in Python.

## Solution Notebook

The solution notebook is named `DS_ML_Coding_Challenge.ipynb`. It contains the following steps:

1. Preprocessing and Feature Engineering
2. Exploratory Data Analysis (EDA)
3. Forecasting the Sourcing Cost
4. Forecasting the Quantity

The notebook uses Python libraries such as pandas, numpy, matplotlib, seaborn, and scikit-learn to perform data preprocessing, EDA, and machine learning.

## Models

The notebook uses the following machine learning models:

1. LSTM Model
2. XGBoost Regressor
3. Random Forest Regressor

The models are trained on the training set and used to predict the sourcing cost and quantity for the test set. The performance of the models is evaluated using mean squared error.

## Results

The notebook provides the following results:

1. Forecasted sourcing cost for June 2021 using LSTM, XGBoost, and Random Forest models.
2. Forecasted quantity for June 2021 using LSTM, XGBoost, and Random Forest models.

The results are presented in tabular form, showing the original and predicted values side by side.

## Conclusion

The notebook concludes by comparing the performance of the three models and selecting the best one based on the mean squared error. The final approach is explained in detail, along with the reasons for choosing it.

## Repository Structure

The repository contains the following files and folders:

1. `DS_ML_Coding_Challenge.ipynb` - Jupyter notebook containing the solution.
2. `DS_ML Coding Challenge Dataset.xlsx` - Dataset provided for the challenge.
3. `train_df_quantify.csv` - Preprocessed dataset used for training the models.
4. `README.md` - This file.

## Requirements

The following Python libraries are required to run the notebook:

1. pandas
2. numpy
3. matplotlib
4. seaborn
5. scikit-learn
6. xgboost
7. keras
8. tensorflow

You can install the required libraries using pip:
```
pip install pandas numpy matplotlib seaborn scikit-learn xgboost keras tensorflow
```
## Usage

1. Clone the repository:
```bash
git clone https://github.com/your-username/DS_ML_Coding_Challenge.git
```
2. Install the required libraries:
```
pip install pandas numpy matplotlib seaborn scikit-learn xgboost keras tensorflow
```
3. Open the Jupyter notebook:
```bash
jupyter notebook DS_ML_Coding_Challenge.ipynb
```
4. Run the notebook and follow the instructions.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvement, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.

## Acknowledgements

Thanks to the company for providing the DS_ML Coding Challenge and the opportunity to work on this interesting problem.





