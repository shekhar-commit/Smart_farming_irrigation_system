import numpy as np

# Stage water need factor
stage_water_need = {
    'germination': 0.8,
    'seedling': 0.7,
    'vegetative': 0.6,
    'flowering': 1.0,
    'fruiting': 0.9,
    'maturity': 0.3
}

# Soil retention factor
soil_retention = {
    'sandy': 0.3,
    'loamy': 0.7,
    'clay': 0.9,
    'silt': 0.6,
    'black': 0.8,
    'red': 0.5,
    'alluvial': 0.75
}

# Crop water requirement by stage (mm)
crop_stage_water_requirement = {
    'germination': 12,
    'seedling': 15,
    'vegetative': 18,
    'flowering': 25,
    'fruiting': 22,
    'maturity': 10
}

def determine_crop_stage(age):
    if age <= 10:
        return 'germination'
    elif age <= 25:
        return 'seedling'
    elif age <= 50:
        return 'vegetative'
    elif age <= 80:
        return 'flowering'
    elif age <= 110:
        return 'fruiting'
    else:
        return 'maturity'


def calculate_soil_saturation_level(soil_moisture, rainfall_last_48h_mm, irrigation_done_today):
    saturation = soil_moisture + (rainfall_last_48h_mm * 0.8) + (8 if irrigation_done_today == 1 else 0)
    return round(min(100, saturation), 2)


def calculate_waterlogging_risk(soil_saturation_level, drainage_condition, soil_type):
    risk_score = 0

    if soil_saturation_level > 85:
        risk_score += 3
    elif soil_saturation_level > 70:
        risk_score += 2
    elif soil_saturation_level > 55:
        risk_score += 1

    if drainage_condition == 'poor':
        risk_score += 2
    elif drainage_condition == 'moderate':
        risk_score += 1

    if soil_type in ['clay', 'black']:
        risk_score += 1

    if risk_score >= 5:
        return 'high'
    elif risk_score >= 3:
        return 'medium'
    else:
        return 'low'


def determine_next_irrigation_action(
    rain_probability_next_24h,
    irrigation_done_today,
    soil_moisture,
    soil_saturation_level,
    waterlogging_risk
):
    if rain_probability_next_24h > 75:
        return "delay"

    if irrigation_done_today == 1 and soil_saturation_level > 60:
        return "skip"

    if waterlogging_risk == "high":
        return "skip"

    if soil_moisture < 25:
        return "irrigate"

    return "monitor"


def calculate_crop_water_requirement(stage):
    return crop_stage_water_requirement.get(stage, 18)


def build_model_input(user_input):
    """
    Farmer ke basic input ko full ML model input me convert karega.
    """

    crop_age_days = int(user_input['crop_age_days'])
    soil_moisture = float(user_input['soil_moisture'])
    rainfall_last_48h_mm = float(user_input['rainfall_last_48h_mm'])
    irrigation_done_today = int(user_input['irrigation_done_today'])
    rain_probability_next_24h = int(user_input['rain_probability_next_24h'])

    crop_stage = determine_crop_stage(crop_age_days)

    soil_saturation_level = calculate_soil_saturation_level(
        soil_moisture,
        rainfall_last_48h_mm,
        irrigation_done_today
    )

    waterlogging_risk = calculate_waterlogging_risk(
        soil_saturation_level,
        user_input['drainage_condition'],
        user_input['soil_type']
    )

    next_irrigation_action = determine_next_irrigation_action(
        rain_probability_next_24h,
        irrigation_done_today,
        soil_moisture,
        soil_saturation_level,
        waterlogging_risk
    )

    crop_water_requirement_mm = calculate_crop_water_requirement(crop_stage)

    final_input = {
        'soil_moisture': soil_moisture,
        'crop_type': user_input['crop_type'],
        'crop_age_days': crop_age_days,
        'crop_stage': crop_stage,
        'soil_type': user_input['soil_type'],
        'temperature_c': float(user_input['temperature_c']),
        'humidity_percent': float(user_input['humidity_percent']),
        'rain_probability_next_24h': rain_probability_next_24h,
        'last_irrigation_days': int(user_input['last_irrigation_days']),
        'soil_temperature_c': float(user_input['soil_temperature_c']),
        'wind_speed_kmph': float(user_input['wind_speed_kmph']),
        'irrigation_method': user_input['irrigation_method'],
        'water_source_type': user_input['water_source_type'],
        'drainage_condition': user_input['drainage_condition'],
        'mulching': user_input['mulching'],
        'rainfall_last_48h_mm': rainfall_last_48h_mm,
        'irrigation_done_today': irrigation_done_today,
        'soil_saturation_level': soil_saturation_level,
        'waterlogging_risk': waterlogging_risk,
        'next_irrigation_action': next_irrigation_action,
        'crop_water_requirement_mm': crop_water_requirement_mm,
        'field_size_acre': float(user_input['field_size_acre']),
        'evapotranspiration_mm_day': float(user_input['evapotranspiration_mm_day']),
        'sunlight_hours': float(user_input['sunlight_hours']),
        'fertilizer_level': user_input['fertilizer_level'],
        'slope_level': user_input['slope_level'],
        'weather_condition': user_input['weather_condition']
    }

    return final_input