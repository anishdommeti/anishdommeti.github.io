from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import pickle
import numpy as np
import datetime
import random
import requests
import logging
from logging.handlers import RotatingFileHandler
from config import get_config

# Initialize Flask app
app = Flask(__name__)

# Load configuration
config = get_config()
app.config.from_object(config)

# Configure CORS
CORS(app, resources={
    r"/*": {
        "origins": config.CORS_ORIGINS,
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Configure Logging
if not app.debug:
    file_handler = RotatingFileHandler(config.LOG_FILE, maxBytes=10240000, backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(getattr(logging, config.LOG_LEVEL))
    app.logger.addHandler(file_handler)
    app.logger.setLevel(getattr(logging, config.LOG_LEVEL))
    app.logger.info('Agriculture ML Assistant startup')
else:
    logging.basicConfig(level=logging.INFO)
    app.logger.setLevel(logging.INFO)

# Load Model and Encoders
try:
    with open(config.MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
    with open(config.ENCODERS_FILE, 'rb') as f:
        encoders = pickle.load(f)
        scaler = encoders.get('scaler')
    app.logger.info("Model and encoders loaded successfully.")
    print("Model and encoders loaded successfully.")
except Exception as e:
    app.logger.error(f"Error loading model: {e}")
    print(f"Error loading model: {e}")
    model = None
    encoders = None
    scaler = None

# Load Monthly Rainfall Averages
try:
    rainfall_df = pd.read_csv(config.RAINFALL_DATA_FILE)
    rainfall_data = rainfall_df.set_index('District').to_dict('index')
    app.logger.info("Rainfall data loaded successfully.")
    print("Rainfall data loaded successfully.")
except Exception as e:
    app.logger.error(f"Error loading rainfall data: {e}")
    print(f"Error loading rainfall data: {e}")
    rainfall_data = {}

# Cost Map (from config)
COST_MAP = config.COST_MAP

def get_estimated_cost(crop, area):
    base = 30000
    for key in COST_MAP:
        if key.lower() in str(crop).lower():
            base = COST_MAP[key]
            break
    return area * base

@app.route('/info', methods=['GET'])
def get_info():
    if not encoders:
        return jsonify({"error": "Model not loaded"}), 500
    
    return jsonify({
        "districts": list(encoders['district'].classes_),
        "seasons": list(encoders['season'].classes_),
        "crops": list(encoders['crop'].classes_)
    })

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not encoders:
        return jsonify({"error": "Model not loaded"}), 500
    
    data = request.json
    district = data.get('district')
    season = data.get('season')
    crop = data.get('crop')
    area = float(data.get('area'))
    rainfall = float(data.get('rainfall')) 
    
    # Calculate Cost if not provided (User might not know exact cost)
    # But for prediction, we need it as feature.
    cost = data.get('cost')
    if not cost:
        cost = get_estimated_cost(crop, area)
    
    try:
        district_enc = encoders['district'].transform([district])[0]
        season_enc = encoders['season'].transform([season])[0]
        crop_enc = encoders['crop'].transform([crop])[0]
        
        # Scale numerical features
        numerical_features = np.array([[area, rainfall, cost]])
        numerical_features_scaled = scaler.transform(numerical_features)
        
        # Combine features: District, Season, Crop, Area_Scaled, Rainfall_Scaled, Cost_Scaled
        features = np.array([[district_enc, season_enc, crop_enc, 
                              numerical_features_scaled[0][0], 
                              numerical_features_scaled[0][1], 
                              numerical_features_scaled[0][2]]])
                              
        prediction_log = model.predict(features)[0]
        prediction = np.expm1(prediction_log)
        
        return jsonify({
            "production": prediction,
            "yield": prediction / area if area > 0 else 0,
            "estimated_cost": cost
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/forecast', methods=['POST'])
def forecast():
    data = request.json
    district = data.get('district')
    current_month_idx = data.get('month_idx', datetime.datetime.now().month - 1)
    
    if district not in rainfall_data:
        return jsonify({"error": "District rainfall data not found"}), 404
        
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    forecast_data = []
    
    for i in range(3):
        idx = (current_month_idx + i) % 12
        month_name = months[idx]
        predicted_rainfall = rainfall_data[district].get(month_name, 0)
        
        erosion_risk = "Low"
        if predicted_rainfall > 300:
            erosion_risk = "High"
        elif predicted_rainfall > 100:
            erosion_risk = "Medium"
            
        forecast_data.append({
            "month": month_name,
            "rainfall": predicted_rainfall,
            "erosion_risk": erosion_risk
        })
        
    high_risk_months = [f['month'] for f in forecast_data if f['erosion_risk'] == 'High']
    advisory = "Conditions are favorable for sowing."
    if high_risk_months:
        advisory = f"Caution: High erosion risk detected in {', '.join(high_risk_months)}. Consider soil conservation measures."
    elif forecast_data[0]['rainfall'] < 50:
         advisory = "Low rainfall expected. Ensure adequate irrigation."

    return jsonify({
        "forecast": forecast_data,
        "advisory": advisory
    })

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    district = data.get('district')
    area = float(data.get('area'))
    budget = data.get('budget') # Low, Medium, High
    slope = data.get('slope') # Flat, Gentle, Steep
    
    # Logic:
    # 1. Filter crops based on Budget
    # 2. Filter based on Slope suitability (Rule based)
    # 3. Predict production for candidates
    # 4. Rank
    
    budget_limit = {
        'Low': config.BUDGET_LIMITS['Low'] * area,
        'Medium': config.BUDGET_LIMITS['Medium'] * area,
        'High': config.BUDGET_LIMITS['High'] * area
    }
    limit = budget_limit.get(budget, config.BUDGET_LIMITS['High'] * area)
    
    # Slope Rules (from config)
    slope_suitability = config.SLOPE_SUITABILITY
    
    # Get all crops
    all_crops = list(encoders['crop'].classes_)
    candidates = []
    
    for crop in all_crops:
        est_cost = get_estimated_cost(crop, area)
        
        # Budget Check
        if est_cost > limit:
            continue
            
        # Slope Check (Soft check - boost score if suitable, don't strictly exclude unless critical)
        # For simplicity, let's just categorize.
        is_suitable_slope = False
        if slope == 'Steep':
            # Check if crop is in steep list OR if it's a general crop that can handle it
            # Let's use a simple keyword match or list
            if any(c in crop for c in slope_suitability['Steep']):
                is_suitable_slope = True
        elif slope == 'Flat':
             if any(c in crop for c in slope_suitability['Flat']):
                is_suitable_slope = True
        else:
            is_suitable_slope = True # Gentle is okay for most
            
        if is_suitable_slope:
            # Predict potential production
            # We need to construct input for prediction
            # District, Season (assume Kharif for now or take input), Crop, Area, Rainfall (avg), Cost
            # This is complex to do in real-time for all crops. 
            # For now, just add to candidates list
            candidates.append(crop)

    return jsonify({
        "recommendations": candidates[:5] # Top 5
    })

def get_weather_forecast(lat, lon):
    try:
        # Open-Meteo API (Free, no key)
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=temperature_2m_max,precipitation_sum,wind_speed_10m_max&timezone=auto"
        response = requests.get(url)
        data = response.json()
        return data
    except Exception as e:
        print(f"Error fetching weather: {e}")
        return None

@app.route('/advisory', methods=['POST'])
def advisory():
    data = request.json
    lat = data.get('lat')
    lon = data.get('lon')
    crop = data.get('crop')
    sowing_date = data.get('sowing_date') # String YYYY-MM-DD
    
    if not lat or not lon:
        return jsonify({"error": "Location required"}), 400
        
    weather = get_weather_forecast(lat, lon)
    if not weather:
        return jsonify({"error": "Could not fetch weather data"}), 500
        
    daily = weather.get('daily', {})
    dates = daily.get('time', [])
    temps = daily.get('temperature_2m_max', [])
    rains = daily.get('precipitation_sum', [])
    winds = daily.get('wind_speed_10m_max', [])
    
    forecast_list = []
    alerts = []
    
    for i in range(len(dates)):
        day_data = {
            "date": dates[i],
            "temp": temps[i],
            "rain": rains[i],
            "wind": winds[i]
        }
        forecast_list.append(day_data)
        
        # Rule Engine for Alerts
        if rains[i] > 50:
            alerts.append(f"⚠️ Heavy rain ({rains[i]}mm) predicted on {dates[i]}. Ensure drainage.")
        if winds[i] > 30:
            alerts.append(f"⚠️ High wind ({winds[i]}km/h) predicted on {dates[i]}. Support tall crops.")
        if temps[i] > 35:
            alerts.append(f"⚠️ High heat ({temps[i]}°C) on {dates[i]}. Irrigate to cool soil.")
            
    # Crop Specific Logic (Mock)
    crop_advice = ""
    if crop:
        if "Rice" in crop:
            if any(r > 20 for r in rains):
                crop_advice = "Rice benefits from this rain, but ensure water level doesn't exceed 5cm."
            else:
                crop_advice = "Dry spell ahead. Maintain standing water for Rice."
        elif "Maize" in crop:
            if any(r > 50 for r in rains):
                crop_advice = "Maize is sensitive to waterlogging. Clear drainage channels immediately!"
    
    return jsonify({
        "forecast": forecast_list,
        "alerts": alerts,
        "crop_advice": crop_advice
    })
@app.route('/payment', methods=['POST'])
def payment():
    data = request.json
    crop = data.get('crop', '')
    
    markets = []
    pests = []
    fertilizer_schedule = []

    if "Turmeric" in crop:
        markets = [
            {"name": "Mowkaiaw Market", "price": "₹15,000/q", "trend": "up"},
            {"name": "Jowai Market", "price": "₹14,500/q", "trend": "up"},
            {"name": "Guwahati Wholesale", "price": "₹18,000/q", "trend": "stable"}
        ]
        pests = [
            {"name": "Rhizome Fly", "risk": "High", "reason": "High moisture predicted in July."},
            {"name": "Leaf Spot", "risk": "Medium", "reason": "Cloudy weather expected."}
        ]
        fertilizer_schedule = [
            {"time": "Basal (At planting)", "action": "FYM 10t/ha + Neem Cake"},
            {"time": "45 Days", "action": "Earthing up + Bio-fertilizer"},
            {"time": "90 Days", "action": "Potash top dressing"}
        ]
    elif "Rice" in crop:
        markets = [
            {"name": "Bara Bazar (Shillong)", "price": "₹3,200/q", "trend": "stable"},
            {"name": "Tura Supermarket", "price": "₹3,000/q", "trend": "down"}
        ]
        pests = [
            {"name": "Stem Borer", "risk": "High", "reason": "Temperature rise in June favors larvae."},
            {"name": "Blast Disease", "risk": "Low", "reason": "Humidity levels are optimal."}
        ]
        fertilizer_schedule = [
            {"time": "Basal", "action": "DAP + MOP (50:50)"},
            {"time": "Tillering (21 Days)", "action": "Urea Top Dressing"},
            {"time": "Panicle Init (60 Days)", "action": "Urea + Potash"}
        ]
    else:
        # Generic/Vegetables
        markets = [
            {"name": "Local Mandi", "price": "₹4,000/q", "trend": "up"},
            {"name": "Direct to Hotel", "price": "₹5,500/q", "trend": "up"}
        ]
        pests = [
            {"name": "Aphids", "risk": "Medium", "reason": "Dry spell favors sucking pests."}
        ]
        fertilizer_schedule = [
            {"time": "Sowing", "action": "Organic Compost"},
            {"time": "Flowering", "action": "Liquid Bio-fertilizer spray"}
        ]

    return jsonify({
        "markets": markets,
        "pests": pests,
        "fertilizerSchedule": fertilizer_schedule
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)