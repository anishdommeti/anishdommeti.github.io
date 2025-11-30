"""
Configuration module for the Agriculture ML Assistant Backend
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Base configuration"""
    
    # Flask Configuration
    FLASK_APP = os.getenv('FLASK_APP', 'app.py')
    FLASK_ENV = os.getenv('FLASK_ENV', 'development')
    DEBUG = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    
    # Server Configuration
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 5000))
    
    # CORS Configuration
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', 'http://localhost:3000,http://localhost:5173').split(',')
    
    # Model Files
    MODEL_FILE = os.getenv('MODEL_FILE', 'agriculture_model_improved.pkl')
    ENCODERS_FILE = os.getenv('ENCODERS_FILE', 'label_encoders_improved.pkl')
    RAINFALL_DATA_FILE = os.getenv('RAINFALL_DATA_FILE', 'rainfall_monthly_averages.csv')
    
    # API Configuration
    WEATHER_API_URL = os.getenv('WEATHER_API_URL', 'https://api.open-meteo.com/v1/forecast')
    
    # Logging Configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'app.log')
    
    # Cost Map for different crops (in ₹ per hectare)
    COST_MAP = {
        'Rice': 40000, 'Banana': 100000, 'Maize': 30000, 'Linseed': 20000,
        'Cowpea(Lobia)': 25000, 'Peas & beans (Pulses)': 35000, 'Rapeseed &Mustard': 25000,
        'Sugarcane': 80000, 'Tobacco': 50000, 'Wheat': 35000, 'Jute': 45000,
        'Mesta': 20000, 'Potato': 60000, 'Turmeric': 70000, 'Ginger': 80000,
        'Arecanut': 90000, 'Black pepper': 100000, 'Cashewnut': 50000, 'Tapioca': 40000
    }
    
    # Budget Limits
    BUDGET_LIMITS = {
        'Low': 20000,
        'Medium': 50000,
        'High': 1000000
    }
    
    # Slope Suitability
    SLOPE_SUITABILITY = {
        'Flat': ['Rice', 'Wheat', 'Jute', 'Potato', 'Sugarcane'],
        'Gentle': ['Maize', 'Soyabean', 'Pulses', 'Vegetables'],
        'Steep': ['Tea', 'Coffee', 'Rubber', 'Arecanut', 'Black pepper', 'Cashewnut', 'Turmeric', 'Ginger']
    }

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    # Override with production values
    FLASK_ENV = 'production'

class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    TESTING = True

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

def get_config(env=None):
    """Get configuration based on environment"""
    if env is None:
        env = os.getenv('FLASK_ENV', 'development')
    return config.get(env, config['default'])
