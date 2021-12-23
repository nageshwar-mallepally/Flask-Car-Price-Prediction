import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from datetime import date

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    form = request.form
    kms = form.get("kms")
    mileage = form.get("mileage")
    engine = form.get("engine_cc")
    power = form.get("max_power")
    seats = form.get("seats")
    car_model = form.get("years")
    years = int(date.today().year)-int(car_model)
    fuel = form.get("fuel")
    seller_type = form.get("seller_type")
    transmission = form.get("transmission")
    owner = form.get("owner")

    fuel_Petrol = 0
    fuel_Diesel = 0
    fuel_LPG = 0
    if(fuel == 'petrol'): fuel_Petrol = 1
    if(fuel == 'diesel'): fuel_Diesel = 1
    if(fuel == 'lpg'): fuel_LPG = 1

    seller_type_individual = 0
    seller_type_trustmark = 0
    if(seller_type == 'individual'): seller_type_individual = 1
    if(seller_type == 'trustmark'): seller_type_trustmark = 1

    transmission_auto = 0
    transmission_manual = 0
    if(transmission == 'auto'): transmission_auto = 1
    if(transmission == 'manual'): transmission_manual = 1

    owner_first = 0
    owner_second = 0
    owner_third = 0
    owner_fourth = 0
    if(owner == 'first'): owner_first = 1
    if(owner == 'second'): owner_second = 1
    if(owner == 'third'): owner_third = 1
    if(owner == 'fourth'): owner_fourth = 1

    inputValues = [[kms, mileage, engine, power, seats, years, fuel_Diesel, fuel_LPG, fuel_Petrol, seller_type_individual, seller_type_trustmark, transmission_manual, owner_fourth, owner_second, owner_first, owner_third ]]
    print(inputValues)
    print(model)
    prediction = model.predict(inputValues)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Expected Price {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
