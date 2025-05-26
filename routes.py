import os
import logging
from flask import render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import torch
from app import app
from model_utils import load_models, predict_fruit, preprocess_image

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# Fruit information database
FRUIT_INFO = {
    'banana': {
        'name': 'Banana',
        'scientific_name': 'Musa',
        'description': 'Bananas are curved yellow fruits with sweet, soft flesh inside. They are one of the most popular fruits worldwide and are rich in potassium, vitamin B6, and vitamin C.',
        'nutritional_benefits': [
            'High in potassium for heart health',
            'Rich in vitamin B6 for brain function',
            'Contains fiber for digestive health',
            'Natural sugars for quick energy',
            'Antioxidants for immune support'
        ],
        'calories_per_100g': 89,
        'season': 'Year-round availability',
        'origin': 'Southeast Asia',
        'storage_tips': 'Store at room temperature until ripe, then refrigerate to slow ripening'
    },
    'watermelon': {
        'name': 'Watermelon',
        'scientific_name': 'Citrullus lanatus',
        'description': 'Watermelons are large, round or oval fruits with green rinds and red, juicy flesh. They are about 92% water and are perfect for hot summer days.',
        'nutritional_benefits': [
            'High water content for hydration',
            'Rich in lycopene, a powerful antioxidant',
            'Contains vitamin C for immune health',
            'Low in calories but high in nutrients',
            'Contains citrulline for heart health'
        ],
        'calories_per_100g': 30,
        'season': 'Summer (June to September)',
        'origin': 'Africa',
        'storage_tips': 'Store whole watermelons at room temperature. Cut watermelon should be refrigerated and consumed within 3-5 days'
    },
    'grapes': {
        'name': 'Grapes',
        'scientific_name': 'Vitis vinifera',
        'description': 'Grapes are small, round or oval fruits that grow in clusters. They come in various colors including green, red, purple, and black, and can be eaten fresh or used to make wine.',
        'nutritional_benefits': [
            'Rich in antioxidants, especially resveratrol',
            'Contains vitamin K for bone health',
            'High in vitamin C for immune support',
            'Natural sugars for energy',
            'Anti-inflammatory compounds'
        ],
        'calories_per_100g': 62,
        'season': 'Late summer to early fall',
        'origin': 'Middle East',
        'storage_tips': 'Store in refrigerator in perforated plastic bags. Do not wash until ready to eat'
    },
    'apple': {
        'name': 'Apple',
        'scientific_name': 'Malus domestica',
        'description': 'Apples are round fruits with crisp flesh and come in many varieties with different colors, flavors, and textures. They are one of the most widely cultivated fruits.',
        'nutritional_benefits': [
            'High in fiber, especially pectin',
            'Rich in antioxidants and flavonoids',
            'Contains vitamin C for immune health',
            'May help regulate blood sugar',
            'Supports heart health'
        ],
        'calories_per_100g': 52,
        'season': 'Fall (September to November)',
        'origin': 'Central Asia',
        'storage_tips': 'Store in refrigerator crisper drawer. Can be stored at room temperature for short periods'
    }
}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Home page with upload form"""
    return render_template('index.html')

@app.route('/classify', methods=['GET', 'POST'])
def classify():
    """Handle image upload and classification"""
    if request.method == 'POST':
        # Check if file was uploaded
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(url_for('index'))
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(url_for('index'))
        
        # Check if file is allowed
        if not allowed_file(file.filename):
            flash('Invalid file type. Please upload an image file.', 'error')
            return redirect(url_for('index'))
        
        try:
            # Secure filename and save
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Load and run prediction
            models = load_models()
            if not models:
                flash('Classification models not available. Please ensure .pth files are in the models directory.', 'error')
                return redirect(url_for('index'))
            
            # Preprocess image and predict
            image = preprocess_image(filepath)
            predictions = predict_fruit(models, image)
            
            # Get the best prediction
            best_prediction = max(predictions, key=lambda x: x['confidence'])
            fruit_type = best_prediction['class']
            confidence = best_prediction['confidence']
            
            # Clean up uploaded file
            os.remove(filepath)
            
            return render_template('classify.html', 
                                 fruit_type=fruit_type,
                                 confidence=confidence,
                                 predictions=predictions,
                                 fruit_info=FRUIT_INFO.get(fruit_type, {}))
        
        except Exception as e:
            logging.error(f"Classification error: {str(e)}")
            flash(f'Error processing image: {str(e)}', 'error')
            return redirect(url_for('index'))
    
    return redirect(url_for('index'))

@app.route('/fruit-info/<fruit_type>')
def fruit_info(fruit_type):
    """Display detailed information about a specific fruit"""
    if fruit_type not in FRUIT_INFO:
        flash('Fruit information not found', 'error')
        return redirect(url_for('index'))
    
    return render_template('fruit_info.html', 
                         fruit_type=fruit_type,
                         fruit_info=FRUIT_INFO[fruit_type])

@app.route('/all-fruits')
def all_fruits():
    """Display information about all supported fruits"""
    return render_template('fruit_info.html', 
                         fruit_type='all',
                         all_fruits=FRUIT_INFO)

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    flash('File is too large. Please upload an image smaller than 16MB.', 'error')
    return redirect(url_for('index'))

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return render_template('index.html'), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors"""
    logging.error(f"Server error: {str(e)}")
    flash('An internal error occurred. Please try again.', 'error')
    return render_template('index.html'), 500
