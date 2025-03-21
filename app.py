import os
import pickle
import torch
import numpy as np
import joblib
from datetime import datetime
import io
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, request, render_template, jsonify, send_file
from werkzeug.utils import secure_filename
from model.model_utils import BirdDroneClassifier, preprocess_data

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['REPORTS_FOLDER'] = 'reports/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.config['ALLOWED_EXTENSIONS'] = {'pkl'}

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['REPORTS_FOLDER'], exist_ok=True)

# Load model and scaler
MODEL_PATH = 'model/model.pth'
SCALER_PATH = 'model/scaler.joblib'

# Get input dimension from model architecture
INPUT_DIM = 10  # Set this to match your feature count

# Load model
model = BirdDroneClassifier(INPUT_DIM)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# Load scaler
scaler = joblib.load(SCALER_PATH)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def generate_report(prediction_result, filename, confidence_score=None):
    """Generate a detailed report based on prediction results"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = {
        "timestamp": now,
        "filename": filename,
        "prediction": prediction_result,
        "confidence_score": confidence_score if confidence_score else "Not available",
        "report_id": f"RPT-{now.replace(' ', '-').replace(':', '')}"
    }
    
    # Create a more detailed PDF report
    from fpdf import FPDF
    
    pdf = FPDF()
    pdf.add_page()
    
    # Add header
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(190, 10, 'Bird/Drone Classification Report', 0, 1, 'C')
    
    # Add report details
    pdf.set_font('Arial', '', 12)
    pdf.cell(190, 10, f"Report ID: {report['report_id']}", 0, 1)
    pdf.cell(190, 10, f"Date: {report['timestamp']}", 0, 1)
    pdf.cell(190, 10, f"Analyzed File: {report['filename']}", 0, 1)
    pdf.cell(190, 10, '', 0, 1)  # Empty line
    
    # Add results section
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(190, 10, 'Classification Results', 0, 1)
    
    pdf.set_font('Arial', '', 12)
    pdf.cell(190, 10, f"Prediction: {report['prediction']}", 0, 1)
    pdf.cell(190, 10, f"Confidence Score: {report['confidence_score']}", 0, 1)
    
    # Add recommendation section
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(190, 10, 'Recommendations', 0, 1)
    
    pdf.set_font('Arial', '', 12)
    if prediction_result == "Bird":
        pdf.multi_cell(190, 10, "The object detected is classified as a bird. No further action needed for airspace security.")
    else:
        pdf.multi_cell(190, 10, "The object detected is classified as a drone. Consider monitoring this object as it may represent an unauthorized entry in restricted airspace.")
    
    # Save the PDF to a bytes buffer
    pdf_buffer = io.BytesIO()
    pdf.output(pdf_buffer)
    pdf_buffer.seek(0)
    
    # Also save to disk for record keeping
    report_path = os.path.join(app.config['REPORTS_FOLDER'], f"{report['report_id']}.pdf")
    with open(report_path, 'wb') as f:
        f.write(pdf_buffer.getbuffer())
    
    # Reset buffer position
    pdf_buffer.seek(0)
    
    return pdf_buffer, report['report_id']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Load the uploaded pickle file
            with open(filepath, 'rb') as f:
                uploaded_data = pickle.load(f)
            
            # Preprocess the data
            processed_data = preprocess_data(uploaded_data, scaler)
            
            # Convert to PyTorch tensor
            input_tensor = torch.FloatTensor(processed_data)
            
            # Make prediction
            with torch.no_grad():
                output = model(input_tensor)
                prediction_value = output.item()
                prediction_label = "Bird" if prediction_value < 0.5 else "Drone"
                confidence_score = (1 - prediction_value) * 100 if prediction_value < 0.5 else prediction_value * 100
            
            # Generate report
            report_buffer, report_id = generate_report(
                prediction_label, 
                filename,
                f"{confidence_score:.2f}%"
            )
            
            # Return success response with prediction
            return jsonify({
                'success': True,
                'prediction': prediction_label,
                'confidence': f"{confidence_score:.2f}%",
                'report_id': report_id
            })
            
        except Exception as e:
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/download_report/<report_id>', methods=['GET'])
def download_report(report_id):
    # Check if report exists in reports folder
    report_path = os.path.join(app.config['REPORTS_FOLDER'], f"{report_id}.pdf")
    
    if os.path.exists(report_path):
        return send_file(
            report_path,
            as_attachment=True,
            download_name=f"classification_report_{report_id}.pdf",
            mimetype='application/pdf'
        )
    
    # If not found, regenerate based on parameters
    filename = request.args.get('filename', 'unknown.pkl')
    prediction = request.args.get('prediction', 'Unknown')
    confidence = request.args.get('confidence', 'Not available')
    
    report_buffer, _ = generate_report(prediction, filename, confidence)
    
    return send_file(
        report_buffer,
        as_attachment=True,
        download_name=f"classification_report_{report_id}.pdf",
        mimetype='application/pdf'
    )

if __name__ == '__main__':
    app.run(debug=True)
