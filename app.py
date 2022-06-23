from statistics import mode
from xgboost import XGBRegressor
import xgboost
import numpy as np
import pandas as pd
import pickle
import sklearn
from flask import Flask, render_template, request
# request for getting dat from Form

# df = pickle.load(open('dataframe.pkl','rb'))
df = pd.read_csv('dataframe.csv')
pipe = pickle.load(open('pipe.pkl', 'rb'))  # I have loaded DataFrame

# #___________________________ model ___________________#
# from sklearn.preprocessing import QuantileTransformer, RobustScaler
# from xgboost import XGBRegressor
# from sklearn.compose import  ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
# from sklearn.pipeline import make_pipeline
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# X = df.copy()
# y = pd.read_csv('label.csv')
# # X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)



# transformer = ColumnTransformer(transformers=[
#     ('scaler1',QuantileTransformer(),['Mileage', 'Engine Capacity', 'Model Year']),
#     ('Ordinal1', OrdinalEncoder(categories=[['Hatchback','Sedan','Suv','Mini Van','Crossover','Van']]),['Body Type']),
#     ('Ordinal2', OrdinalEncoder(categories=[['Petrol','Diesel','Hybrid']]), ['Engine Type']),
#     ('Ordinal3', OrdinalEncoder(categories=[['Automatic', 'Manual']]), ['Transmission']),
#     ('ohe1', (OneHotEncoder(drop='first', sparse=False)),['Model','variant','Color','Manufacturer', 'Assembly'])   
# ], remainder='passthrough')

# pipe_r = make_pipeline(transformer, RandomForestRegressor())
# pipe_r.fit(X_train,y_train)

# pipe_r.predict(X_test)



app = Flask(__name__)  # important

@app.route('/')  # for pointing homepage
def index():

    model_year = sorted(df['Model Year'].unique())
    mileage = sorted(df['Mileage'].unique())
    engine_type = sorted(df['Engine Type'].unique())
    engine_capacity = sorted(df['Engine Capacity'].unique())
    transmission = sorted(df['Transmission'].unique())
    color = sorted(df['Color'].unique())
    Assembly = sorted(df['Assembly'].unique())
    body_type = sorted(df['Body Type'].unique())
    manufacturer = sorted(df['Manufacturer'].unique())
    model = df['Model'].unique()
    variant = df['variant'].unique()
    abs = df['ABS'].unique()
    fm_radio = df['AM/FM Radio'].unique()
    air_bags = df['Air Bags'].unique()
    air_cond = df['Air Conditioning'].unique()
    allay_rim = df['Alloy Rims'].unique()
    cd_player = df['CD Player'].unique()
    cassette_player = df['Cassette Player'].unique()
    climate_control = df['Climate Control'].unique()
    coolbox = df['CoolBox'].unique()
    cruise_control = df['Cruise Control'].unique()
    dvd_player = df['DVD Player'].unique()
    front_camera = df['Front Camera'].unique()
    front_speakers = df['Front Speakers'].unique()
    heated_seats = df['Heated Seats'].unique()
    immobilizer_key = df['Immobilizer Key'].unique()
    keyless_entry = df['Keyless Entry'].unique()
    navigation_system = df['Navigation System'].unique()
    power_locks = df['Power Locks'].unique()
    power_mirrors = df['Power Mirrors'].unique()
    power_steering = df['Power Steering'].unique()
    power_windows = df['Power Windows'].unique()
    rear_ac = df['Rear AC Vents'].unique()
    rear_camera = df['Rear Camera'].unique()
    rear_seat = df['Rear Seat Entertainment'].unique()
    rear_speakers = df['Rear speakers'].unique()
    steering_switches = df['Steering Switches'].unique()
    sun_roof = df['Sun Roof'].unique()
    usb = df['USB and Auxillary Cable'].unique()

    return render_template('index.html', model_year=model_year, mileage=mileage, engine_type=engine_type, engine_capacity=engine_capacity, transmission=transmission, color=color, Assembly=Assembly, body_type=body_type, manufacturer=manufacturer, model=model, variant=variant, abs=abs, fm_radio=fm_radio, air_bags=air_bags, air_cond=air_cond, allay_rim=allay_rim, cd_player=cd_player, cassette_player=cassette_player, climate_control=climate_control, coolbox=coolbox, cruise_control=cruise_control, dvd_player=dvd_player, front_camera=front_camera, front_speakers=front_speakers, heated_seats=heated_seats, immobilizer_key=immobilizer_key, keyless_entry=keyless_entry, navigation_system=navigation_system, power_locks=power_locks, power_mirrors=power_mirrors, power_steering=power_steering, power_windows=power_windows, rear_ac=rear_ac, rear_camera=rear_camera, rear_seat=rear_seat, rear_speakers=rear_speakers, steering_switches=steering_switches, sun_roof=sun_roof, usb=usb)


@app.route('/predict', methods=['POST'])
def predict():
    model_year = request.form.get('model_year')
    mileage = request.form.get('mileage')
    engine_type = request.form.get('engine_type')
    engine_capacity = request.form.get('engine_capacity')
    transmission = request.form.get('transmission')
    color = request.form.get('color')
    Assembly = request.form.get('Assembly')
    body_type = request.form.get('body_type')
    manufacturer = request.form.get('manufacturer')
    model = request.form.get('model')
    variant = request.form.get('variant')
    abs = request.form.get('abs')
    fm_radio = request.form.get('fm_radio')
    air_bags = request.form.get('air_bags')
    air_cond = request.form.get('air_cond')
    allay_rim = request.form.get('allay_rim')
    cd_player = request.form.get('cd_player')
    cassette_player = request.form.get('cassette_player')
    climate_control = request.form.get('climate_control')
    coolbox = request.form.get('coolbox')
    cruise_control = request.form.get('cruise_control')
    dvd_player = request.form.get('dvd_player')
    front_camera = request.form.get('front_camera')
    front_speakers = request.form.get('front_speakers')
    heated_seats = request.form.get('heated_seats')
    immobilizer_key = request.form.get('immobilizer_key')
    keyless_entry = request.form.get('keyless_entry')
    navigation_system = request.form.get('navigation_system')
    power_locks = request.form.get('power_locks')
    power_mirrors = request.form.get('power_mirrors')
    power_steering = request.form.get('power_steering')
    power_windows = request.form.get('power_windows')
    rear_ac = request.form.get('rear_ac')
    rear_camera = request.form.get('rear_camera')
    rear_seat = request.form.get('rear_seat')
    rear_speakers = request.form.get('rear_speakers')
    steering_switches = request.form.get('steering_switches')
    sun_roof = request.form.get('sun_roof')
    usb = request.form.get('usb')

    input = pd.DataFrame([[model_year, mileage, engine_type, engine_capacity, transmission, color, Assembly, body_type, manufacturer, model, variant, abs, fm_radio, air_bags, air_cond, allay_rim, cd_player, cassette_player, climate_control, coolbox, cruise_control,dvd_player, front_camera, front_speakers, heated_seats, immobilizer_key, keyless_entry, navigation_system, power_locks, power_mirrors, power_steering, power_windows, rear_ac, rear_camera, rear_seat, rear_speakers, steering_switches, sun_roof, usb]])

    prediction = pipe.predict(input.to_xarray())
    return str(prediction)
    # return str(prediction)/
    # return render_template('index.html', prediction_text=f'Predicted Price is {prediction}')

if __name__ == "__main__":
    app.run(debug=True)
