# Machine Learning with MSA Exports
<b> Josh Barker - Final Project DTSC 691 Eastern University </b>

In this project, using data from the International Trade Administration's Metropolitan Export Series (2005 - 2023), I followed export volumes of United States Metropolitan Statistical Areas (MSA) over time. Taking data from other sources, I modeled how factors from general economic conditions to employment in the manufacturing sector, state policies like minimum wage and corporate income taxes, and exogenous factors like weather could impact export volumes. After training several machine learning models on the data, the best models were selected and incorporated into this site.

There are two models:

<b>Changed Conditions</b>: This model seeks to apply the concept of ceteris paribus to the MSA. All else staying the same, if the average temperature increased by 5 degrees or the corporate income tax rate was lower by a percentage point, how would we expect the export volumes to change?

<b>Forecast</b>: This model seeks to forecast the next year's exports based on the same factors. Can we predict how this year's conditions will impact next year's output?

## What's in this repository
The code is broken up into 4 Jupyter Notebooks for data exploration and model training and 1 Jupyter Notebook, a .py initialization file, and 9 html files for a user-facing Flask App.

### Data Exploration and Model Training
* Capstone Notebook 1 - Data Import and Merge.ipynb
* Capstone Notebook 2 - Data Cleaning and Preparation.ipynb
* Capstone Notebook 3 - Exploratory Data Analysis.ipynb
* Capstone Notebook 4 - Model Training.ipynb

### Flask App
* Capstone Notebook 5 - Website Prep.ipynb
* __init__.py
* requirements.txt
* base.html
* home.html
* about.html
* changed_conditions_predictor.html
* prefill_lag.html
* lag_prediction.html
* forecastpredictor.html
* prefill_forecast.html
* forecast_prediction.html
