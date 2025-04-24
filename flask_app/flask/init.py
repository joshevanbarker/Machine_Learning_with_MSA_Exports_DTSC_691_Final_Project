import os
import pandas as pd
import numpy as np
import joblib
from flask import Flask, render_template, request

app = Flask(__name__, instance_relative_config=True, static_folder="templates")
app.config.from_mapping(
    SECRET_KEY='dev',
    DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
)



forecasted_msa_data = pd.read_csv(os.path.join(app.root_path, 'templates', 'forecast_msa_table.csv'))
forecast_msas = list(forecasted_msa_data["MSA"])
lag_msa_data = pd.read_csv(os.path.join(app.root_path, 'templates', 'lag_msa_table.csv'))

@app.route('/home')
def home():
    return render_template("home.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/changed_conditions_predictor')
def changed_conditions_predictor():
    return render_template("changed_conditions_predictor.html", dropdown = forecast_msas)

@app.route('/prefill_lag', methods=['GET', 'POST'])
def prefill_lag():
    if request.method == "POST":
        msa = request.form.get("msa")
        lag_msa_data = pd.read_csv(os.path.join(app.root_path, 'templates', 'lag_msa_table.csv'))
        lag_msa_data = round(lag_msa_data, 2)
        lag_msa_data = lag_msa_data.rename({"S&P500_Close":"SP500_Close", "lag_exports": "lagged_export"}, axis = 1)
        lag_msa_data["Top_Corporate_Income_Tax_Rate"] = forecasted_msa_data["Top_Corporate_Income_Tax_Rate"] * 100
        try:
            MSA_data = lag_msa_data[lag_msa_data["MSA"] == msa].iloc[0].to_dict()
            return render_template('prefill_lag.html', msa = msa, **MSA_data)

        except ValueError:
            return "Please Enter valid values"

        pass
    pass

@app.route('/lag_predictor', methods=['GET', 'POST'])
def lag_predictor():
    if request.method == "POST":
        # get form data
        Per_Capita_Income = request.form.get('Per_Capita_Income')
        Population = request.form.get('Population')
        Manufacturing_Employment = request.form.get('Manufacturing_Employment')
        Top_Corporate_Income_Tax_Rate = request.form.get('Top_Corporate_Income_Tax_Rate')
        FHFA_index = request.form.get('FHFA_index')
        avg_weather = request.form.get('avg_weather')
        energy_consumption = request.form.get('energy_consumption')
        Minimum_Wage = request.form.get('Minimum_Wage')
        SP500_Close = request.form.get('SP500_Close')
        lagged_export = request.form.get('lagged_export')

        # call preprocessDataAndPredict and pass inputs
        try:
            prediction = policychange_lag(Per_Capita_Income, Population, Manufacturing_Employment, Top_Corporate_Income_Tax_Rate, FHFA_index, avg_weather, energy_consumption, Minimum_Wage, SP500_Close, lagged_export)
            # pass prediction to template
            return render_template('lag_prediction.html', prediction = prediction)

        except ValueError:
            return "Please Enter valid values"

        pass
    pass

def policychange_lag(Per_Capita_Income, Population, Manufacturing_Employment, Top_Corporate_Income_Tax_Rate, FHFA_index, avg_weather, energy_consumption, Minimum_Wage, SP500_Close, lagged_export):


    #Create dataframe for prediction
    data = pd.DataFrame({"Per_Capita_Income": [Per_Capita_Income], "Population":[Population], "Manufacturing_Employment":[Manufacturing_Employment],
                         "Top_Corporate_Income_Tax_Rate":[float(Top_Corporate_Income_Tax_Rate)/100], "FHFA_index":[FHFA_index],
                         "avg_weather":[avg_weather], "energy_consumption":[energy_consumption], "Minimum_Wage":[Minimum_Wage],
                         "S&P500_Close":[SP500_Close], "lagged_log_exports":[np.log(float(lagged_export))]})
    #Open the Model
    pickle_file = open('best_lag_model.pkl', "rb")

    # Load trained model using joblib
    trained_model = joblib.load(pickle_file)

    # Use the model to predict
    prediction = trained_model.predict(data)

    #Inverse Log Transform
    prediction = round(float(np.exp(prediction)[0]), 3)

    return prediction

@app.route('/forecast_predictor')
def forecast_predictor():
    forecasted_msa_data = pd.read_csv(os.path.join(app.root_path, 'templates', 'forecast_msa_table.csv'))
    forecast_msas = list(forecasted_msa_data["MSA"])
    return render_template("forecastpredictor.html", dropdown = forecast_msas)

@app.route('/prefill_forecast', methods=['GET', 'POST'])
def prefill_forecast():
    if request.method == "POST":
        msa = request.form.get("msa")
        forecasted_msa_data = pd.read_csv(os.path.join(app.root_path, 'templates', 'forecast_msa_table.csv'))
        forecasted_msa_data = round(forecasted_msa_data, 2)
        forecasted_msa_data = forecasted_msa_data.rename({"S&P500_Close":"SP500_Close"}, axis = 1)
        forecasted_msa_data["Top_Corporate_Income_Tax_Rate"] = forecasted_msa_data["Top_Corporate_Income_Tax_Rate"] * 100
        try:
            MSA_data = forecasted_msa_data[forecasted_msa_data["MSA"] == msa].iloc[0].to_dict()
            return render_template('prefill_forecast.html', msa = msa, **MSA_data)

        except ValueError:
            return "Please Enter valid values"

        pass
    pass

@app.route('/forecasted_predictor', methods=['GET', 'POST'])
def forecasted_predictor():
    if request.method == "POST":
        # get form data
        Per_Capita_Income = request.form.get('Per_Capita_Income')
        Population = request.form.get('Population')
        Manufacturing_Employment = request.form.get('Manufacturing_Employment')
        Top_Corporate_Income_Tax_Rate = request.form.get('Top_Corporate_Income_Tax_Rate')
        FHFA_index = request.form.get('FHFA_index')
        avg_weather = request.form.get('avg_weather')
        energy_consumption = request.form.get('energy_consumption')
        Minimum_Wage = request.form.get('Minimum_Wage')
        SP500_Close = request.form.get('SP500_Close')
        exports = request.form.get('exports')

        # call preprocessDataAndPredict and pass inputs
        try:
            prediction = forecast(Per_Capita_Income, Population, Manufacturing_Employment, Top_Corporate_Income_Tax_Rate, FHFA_index, avg_weather, energy_consumption, Minimum_Wage, SP500_Close, exports)
            # pass prediction to template
            return render_template('forecast_prediction.html', prediction = prediction)

        except ValueError:
            return "Please Enter valid values"

        pass
    pass

def forecast(Per_Capita_Income, Population, Manufacturing_Employment, Top_Corporate_Income_Tax_Rate, FHFA_index, avg_weather, energy_consumption, Minimum_Wage, SP500_Close, exports):


    #Create dataframe for prediction
    data = pd.DataFrame({"Per_Capita_Income": [Per_Capita_Income], "Population":[Population], "Manufacturing_Employment":[Manufacturing_Employment],
                         "Top_Corporate_Income_Tax_Rate":[float(Top_Corporate_Income_Tax_Rate)/100], "FHFA_index":[FHFA_index],
                         "avg_weather":[avg_weather], "energy_consumption":[energy_consumption], "Minimum_Wage":[Minimum_Wage],
                         "S&P500_Close":[SP500_Close], "log_exports":[np.log(float(exports))]})
    #Open the Model
    pickle_file = open('best_forecast_model.pkl', "rb")

    # Load trained model using joblib
    trained_model = joblib.load(pickle_file)

    # Use the model to predict
    prediction = trained_model.predict(data)

    #Inverse Log Transform
    prediction = round(float(np.exp(prediction)[0]), 2)

    return prediction

