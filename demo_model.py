import pickle
import pandas as pd
import numpy as np

def demo_model():
    # Load Model and Encoders
    try:
        with open('agriculture_model_improved.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('label_encoders_improved.pkl', 'rb') as f:
            encoders = pickle.load(f)
            scaler = encoders.get('scaler')
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Define some test cases
    # District, Season, Crop, Area, Rainfall
    # Note: Cost is needed for the model but not in this simple list. We'll estimate it.
    
    # Cost Map (Simplified for demo)
    COST_MAP = {
        'Rice': 40000, 'Banana': 100000, 'Maize': 30000
    }
    
    test_cases = [
        {
            'District': 'EAST KHASI HILLS',
            'Season': 'Kharif',
            'Crop': 'Rice',
            'Area': 5000,
            'Rainfall': 2000 # Normal-ish
        },
        {
            'District': 'EAST KHASI HILLS',
            'Season': 'Kharif',
            'Crop': 'Rice',
            'Area': 5000,
            'Rainfall': 1000 # Low Rainfall (Deficit)
        },
        {
            'District': 'WEST GARO HILLS',
            'Season': 'Whole Year',
            'Crop': 'Banana',
            'Area': 1000,
            'Rainfall': 2500
        }
    ]

    print("\n--- Model Demonstration (Improved Model) ---")
    print(f"{'District':<20} | {'Season':<10} | {'Crop':<10} | {'Area (Ha)':<10} | {'Rainfall (mm)':<15} | {'Pred. Production (Tonnes)':<25}")
    print("-" * 100)

    for case in test_cases:
        try:
            district_enc = encoders['district'].transform([case['District']])[0]
            season_enc = encoders['season'].transform([case['Season']])[0]
            crop_enc = encoders['crop'].transform([case['Crop']])[0]
            
            # Estimate Cost
            base_cost = COST_MAP.get(case['Crop'], 30000)
            cost = case['Area'] * base_cost
            
            # Scale numerical features
            numerical_features = np.array([[case['Area'], case['Rainfall'], cost]])
            numerical_features_scaled = scaler.transform(numerical_features)
            
            features = np.array([[district_enc, season_enc, crop_enc, 
                                  numerical_features_scaled[0][0], 
                                  numerical_features_scaled[0][1], 
                                  numerical_features_scaled[0][2]]])
                                  
            prediction_log = model.predict(features)[0]
            prediction = np.expm1(prediction_log)
            
            print(f"{case['District']:<20} | {case['Season']:<10} | {case['Crop']:<10} | {case['Area']:<10} | {case['Rainfall']:<15} | {prediction:<25.2f}")
        except Exception as e:
            print(f"Error predicting for {case}: {e}")

if __name__ == "__main__":
    demo_model()
