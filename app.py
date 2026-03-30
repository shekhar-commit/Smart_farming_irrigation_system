from flask import Flask, render_template, request
import pandas as pd
import joblib
from utils.feature_engineering import build_model_input

app = Flask(__name__)

# ----------------------------
# Lazy Load Models
# ----------------------------
model1 = None
model2_reg = None
model2_time = None
model3 = None


def load_models():
    global model1, model2_reg, model2_time, model3

    if model1 is None:
        model1 = joblib.load("model1_irrigation_needed.pkl")

    if model2_reg is None:
        model2_reg = joblib.load("model2_water_and_duration.pkl")

    if model2_time is None:
        model2_time = joblib.load("model2_best_irrigation_time.pkl")

    if model3 is None:
        model3 = joblib.load("model3_next_irrigation_days.pkl")


# ----------------------------
# Home Page
# ----------------------------
@app.route("/")
def home():
    return render_template("index.html")


# ----------------------------
# Prediction Route
# ----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        load_models()

        # Raw form input
        user_input = {
            'soil_moisture': request.form['soil_moisture'],
            'crop_type': request.form['crop_type'],
            'crop_age_days': request.form['crop_age_days'],
            'soil_type': request.form['soil_type'],
            'temperature_c': request.form['temperature_c'],
            'humidity_percent': request.form['humidity_percent'],
            'rain_probability_next_24h': request.form['rain_probability_next_24h'],
            'last_irrigation_days': request.form['last_irrigation_days'],
            'soil_temperature_c': request.form['soil_temperature_c'],
            'wind_speed_kmph': request.form['wind_speed_kmph'],
            'irrigation_method': request.form['irrigation_method'],
            'water_source_type': request.form['water_source_type'],
            'drainage_condition': request.form['drainage_condition'],
            'mulching': request.form['mulching'],
            'rainfall_last_48h_mm': request.form['rainfall_last_48h_mm'],
            'irrigation_done_today': request.form['irrigation_done_today'],
            'field_size_acre': request.form['field_size_acre'],
            'evapotranspiration_mm_day': request.form['evapotranspiration_mm_day'],
            'sunlight_hours': request.form['sunlight_hours'],
            'fertilizer_level': request.form['fertilizer_level'],
            'slope_level': request.form['slope_level'],
            'weather_condition': request.form['weather_condition']
        }

        # Build model input
        final_input = build_model_input(user_input)
        input_df = pd.DataFrame([final_input])

        # ----------------------------
        # Predictions
        # ----------------------------
        pred1 = model1.predict(input_df)[0]
        pred1_proba = model1.predict_proba(input_df)[0][1]

        pred2 = model2_reg.predict(input_df)[0]
        pred2_time = model2_time.predict(input_df)[0]
        pred3 = model3.predict(input_df)[0]

        # Human readable irrigation decision
        irrigation_needed_text = "हाँ, अभी सिंचाई करनी चाहिए" if pred1 == 1 else "नहीं, अभी सिंचाई की जरूरत नहीं है"

        # Reasons
        reasons = []

        if final_input['soil_moisture'] < 25:
            reasons.append("मिट्टी की नमी कम है")

        if final_input['crop_stage'] in ['flowering', 'fruiting']:
            reasons.append("फसल अधिक पानी वाली अवस्था में है")

        if final_input['temperature_c'] > 32:
            reasons.append("तापमान ज्यादा है, पानी जल्दी सूख सकता है")

        if final_input['humidity_percent'] < 40:
            reasons.append("हवा में नमी कम है")

        if final_input['soil_type'] == 'sandy':
            reasons.append("रेतीली मिट्टी पानी जल्दी छोड़ती है")

        if final_input['rain_probability_next_24h'] > 70:
            reasons.append("बारिश की संभावना ज्यादा है")

        if final_input['waterlogging_risk'] == 'high':
            reasons.append("पानी भराव का खतरा है")

        if final_input['mulching'] == 'yes':
            reasons.append("मल्चिंग होने से नमी बनी रहती है")

        if len(reasons) == 0:
            reasons.append("खेत की स्थिति सामान्य है")

        # Final result object
        result = {
            "irrigation_needed": irrigation_needed_text,
            "confidence": round(float(pred1_proba * 100), 2),
            "water_needed": round(float(pred2[0]), 2),
            "duration": round(float(pred2[1]), 2),
            "best_time": str(pred2_time),
            "next_days": round(float(pred3), 2),
            "crop_stage": final_input['crop_stage'],
            "soil_saturation_level": round(float(final_input['soil_saturation_level']), 2),
            "waterlogging_risk": final_input['waterlogging_risk'],
            "next_irrigation_action": final_input['next_irrigation_action'],
            "crop_water_requirement_mm": round(float(final_input['crop_water_requirement_mm']), 2),
            "reasons": reasons
        }

        return render_template("result.html", result=result)

    except Exception as e:
        return f"<h2>❌ Error: {str(e)}</h2>"


# ----------------------------
# Run App
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)