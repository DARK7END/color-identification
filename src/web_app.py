#!/usr/bin/env python3
"""
Web Application for Color Detection
Flask-based web interface for color analysis.
"""

from flask import Flask, render_template, request, jsonify, send_file, url_for
from flask_cors import CORS
import os
import uuid
from werkzeug.utils import secure_filename
import json
from PIL import Image
import io
import base64

from color_detector import ColorDetector

app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULTS_FOLDER'] = 'results'

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

# Create directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Global detector instance
detector = ColorDetector()

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_unique_filename(filename):
    """Generate unique filename"""
    ext = filename.rsplit('.', 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    return unique_name

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_image():
    """Analyze uploaded image for colors"""
    try:
        # Check if file was uploaded
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Get parameters
        num_colors = int(request.form.get('num_colors', 5))
        num_colors = max(1, min(20, num_colors))  # Limit between 1-20
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        unique_filename = generate_unique_filename(filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        
        # Analyze image
        results = detector.analyze_image(file_path, num_colors=num_colors, save_results=False)
        
        if not results:
            return jsonify({'error': 'Failed to analyze image'}), 500
        
        # Generate image ID for session
        image_id = unique_filename.split('.')[0]
        
        # Store results temporarily (in production, use database or cache)
        results_path = os.path.join(app.config['RESULTS_FOLDER'], f"{image_id}_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, default=str)
        
        # Create response
        response_data = {
            'image_id': image_id,
            'image_url': url_for('static', filename=f'uploads/{unique_filename}'),
            'dominant_colors': results['dominant_colors'],
            'distribution_stats': results['distribution_stats'],
            'analysis_timestamp': results['analysis_timestamp']
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"[ERROR] Analysis failed: {e}")
        return jsonify({'error': 'Analysis failed'}), 500

@app.route('/palette/<image_id>')
def get_palette(image_id):
    """Get color palette for analyzed image"""
    try:
        # Load results
        results_path = os.path.join(app.config['RESULTS_FOLDER'], f"{image_id}_results.json")
        
        if not os.path.exists(results_path):
            return jsonify({'error': 'Results not found'}), 404
        
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        # Get number of colors from query parameter
        num_colors = int(request.args.get('colors', len(results['dominant_colors'])))
        dominant_colors = results['dominant_colors'][:num_colors]
        
        # Create palette image
        palette_image = detector.create_color_palette(dominant_colors, (400, 100))
        
        # Convert to base64 for web display
        img_buffer = io.BytesIO()
        palette_image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        palette_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        return jsonify({
            'palette_image': f"data:image/png;base64,{palette_base64}",
            'colors': dominant_colors
        })
        
    except Exception as e:
        print(f"[ERROR] Palette generation failed: {e}")
        return jsonify({'error': 'Palette generation failed'}), 500

@app.route('/download/<image_id>')
def download_results(image_id):
    """Download analysis results"""
    try:
        format_type = request.args.get('format', 'json').lower()
        
        # Load results
        results_path = os.path.join(app.config['RESULTS_FOLDER'], f"{image_id}_results.json")
        
        if not os.path.exists(results_path):
            return jsonify({'error': 'Results not found'}), 404
        
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        if format_type == 'json':
            return send_file(results_path, as_attachment=True, 
                           download_name=f"{image_id}_analysis.json")
        
        elif format_type == 'csv':
            # Convert to CSV format
            csv_path = os.path.join(app.config['RESULTS_FOLDER'], f"{image_id}_analysis.csv")
            
            with open(csv_path, 'w') as f:
                f.write("Color_Index,RGB,HEX,Percentage,Color_Name\\n")
                for i, color in enumerate(results['dominant_colors']):
                    rgb_str = f"({color['rgb'][0]},{color['rgb'][1]},{color['rgb'][2]})"
                    f.write(f"{i+1},{rgb_str},{color['hex']},{color['percentage']},{color['color_name']}\\n")
            
            return send_file(csv_path, as_attachment=True, 
                           download_name=f"{image_id}_analysis.csv")
        
        else:
            return jsonify({'error': 'Unsupported format'}), 400
            
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        return jsonify({'error': 'Download failed'}), 500

@app.route('/api/color-info')
def get_color_info():
    """Get detailed color information"""
    try:
        # Get color from query parameters
        hex_color = request.args.get('hex')
        if not hex_color:
            return jsonify({'error': 'No color provided'}), 400
        
        # Convert hex to RGB
        rgb = detector.color_utils.hex_to_rgb(hex_color)
        
        # Get color information
        color_info = {
            'rgb': rgb,
            'hex': hex_color,
            'hsv': detector.color_utils.rgb_to_hsv(rgb),
            'cmyk': detector.color_utils.rgb_to_cmyk(rgb),
            'name': detector.color_utils.get_color_name(rgb),
            'brightness': detector.color_utils.get_color_brightness(rgb),
            'is_dark': detector.color_utils.is_dark_color(rgb),
            'complementary': detector.color_utils.get_complementary_color(rgb),
            'analogous': detector.color_utils.get_analogous_colors(rgb),
            'triadic': detector.color_utils.get_triadic_colors(rgb)
        }
        
        return jsonify(color_info)
        
    except Exception as e:
        print(f"[ERROR] Color info failed: {e}")
        return jsonify({'error': 'Failed to get color information'}), 500

@app.route('/api/stats')
def get_stats():
    """Get application statistics"""
    try:
        # Count uploaded files
        upload_count = len([f for f in os.listdir(app.config['UPLOAD_FOLDER']) 
                           if os.path.isfile(os.path.join(app.config['UPLOAD_FOLDER'], f))])
        
        # Count analysis results
        results_count = len([f for f in os.listdir(app.config['RESULTS_FOLDER']) 
                            if f.endswith('_results.json')])
        
        stats = {
            'total_uploads': upload_count,
            'total_analyses': results_count,
            'supported_formats': list(ALLOWED_EXTENSIONS),
            'max_file_size_mb': app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)
        }
        
        return jsonify(stats)
        
    except Exception as e:
        print(f"[ERROR] Stats failed: {e}")
        return jsonify({'error': 'Failed to get statistics'}), 500

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({'error': 'File too large'}), 413

@app.errorhandler(404)
def not_found(e):
    """Handle not found error"""
    return jsonify({'error': 'Resource not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server error"""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("[INFO] Starting Color Detection Web Application")
    print(f"[INFO] Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"[INFO] Results folder: {app.config['RESULTS_FOLDER']}")
    print(f"[INFO] Max file size: {app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)}MB")
    print(f"[INFO] Supported formats: {', '.join(ALLOWED_EXTENSIONS)}")
    
    # Run the application
    app.run(host='0.0.0.0', port=5000, debug=True)

