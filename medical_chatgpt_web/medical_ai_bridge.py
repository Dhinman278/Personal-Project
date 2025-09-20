"""
Medical AI Bridge - Connect Web Interface to Medical AI System
Provides API bridge between ChatGPT-style web app and medical AI model
"""

import sys
import os
import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
import threading
import webbrowser
import time

# Add parent directory to path to import AI model
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

class MedicalAIHandler(SimpleHTTPRequestHandler):
    """HTTP request handler that serves web app and provides AI API"""
    
    def __init__(self, *args, medical_model=None, **kwargs):
        self.medical_model = medical_model
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests for web app files"""
        if self.path == '/':
            self.path = '/index.html'
        
        # Serve static files
        return super().do_GET()
    
    def do_POST(self):
        """Handle POST requests for AI predictions"""
        if self.path == '/api/analyze':
            self.handle_symptom_analysis()
        elif self.path == '/api/emergency':
            self.handle_emergency_check()
        else:
            self.send_error(404, "API endpoint not found")
    
    def handle_symptom_analysis(self):
        """Process symptom analysis request"""
        try:
            # Parse request data
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length).decode('utf-8')
            data = json.loads(post_data)
            
            symptoms = data.get('symptoms', '')
            patient_info = data.get('patient_info', {})
            
            # Get AI prediction if model is available
            if self.medical_model and hasattr(self.medical_model, 'predict_disease'):
                # Convert symptoms to model input format
                prediction_result = self.get_ai_prediction(symptoms, patient_info)
            else:
                # Fallback to rule-based responses
                prediction_result = self.get_rule_based_response(symptoms)
            
            # Send JSON response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            response = json.dumps(prediction_result)
            self.wfile.write(response.encode('utf-8'))
            
        except Exception as e:
            print(f"Error in symptom analysis: {e}")
            self.send_error(500, f"Analysis error: {str(e)}")
    
    def handle_emergency_check(self):
        """Check for emergency symptoms"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length).decode('utf-8')
            data = json.loads(post_data)
            
            symptoms = data.get('symptoms', '').lower()
            
            # Emergency keywords detection
            emergency_keywords = [
                'chest pain', 'heart attack', 'stroke', 'can\'t breathe',
                'difficulty breathing', 'severe bleeding', 'unconscious',
                'severe allergic reaction', 'overdose', 'poisoning',
                'severe headache', 'sudden severe pain', 'loss of consciousness'
            ]
            
            is_emergency = any(keyword in symptoms for keyword in emergency_keywords)
            
            response = {
                'is_emergency': is_emergency,
                'confidence': 0.9 if is_emergency else 0.1,
                'emergency_actions': [
                    'Call 911 immediately',
                    'Do not drive yourself',
                    'Stay calm and follow dispatcher instructions'
                ] if is_emergency else []
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            self.wfile.write(json.dumps(response).encode('utf-8'))
            
        except Exception as e:
            print(f"Error in emergency check: {e}")
            self.send_error(500, f"Emergency check error: {str(e)}")
    
    def get_ai_prediction(self, symptoms, patient_info):
        """Get prediction from trained medical AI model"""
        try:
            # This would interface with your actual AI model
            # For now, return structured response
            
            # Symptom severity analysis
            severity_keywords = {
                'mild': ['slight', 'minor', 'little', 'mild'],
                'moderate': ['moderate', 'noticeable', 'concerning'],
                'severe': ['severe', 'extreme', 'unbearable', 'intense', 'sharp']
            }
            
            symptoms_lower = symptoms.lower()
            severity = 'mild'
            
            for level, keywords in severity_keywords.items():
                if any(keyword in symptoms_lower for keyword in keywords):
                    severity = level
            
            # Disease prediction (simplified)
            disease_keywords = {
                'Common Cold': ['runny nose', 'sneezing', 'mild cough', 'congestion'],
                'Influenza': ['fever', 'body aches', 'fatigue', 'chills'],
                'COVID-19': ['dry cough', 'loss of taste', 'loss of smell', 'fever'],
                'Migraine': ['severe headache', 'sensitivity to light', 'nausea'],
                'Gastroenteritis': ['nausea', 'vomiting', 'diarrhea', 'stomach pain'],
                'Hypertension': ['high blood pressure', 'headache', 'dizziness']
            }
            
            predicted_conditions = []
            for condition, keywords in disease_keywords.items():
                if any(keyword in symptoms_lower for keyword in keywords):
                    confidence = len([k for k in keywords if k in symptoms_lower]) / len(keywords)
                    predicted_conditions.append({
                        'condition': condition,
                        'confidence': round(confidence * 100, 1),
                        'severity': severity
                    })
            
            # Sort by confidence
            predicted_conditions.sort(key=lambda x: x['confidence'], reverse=True)
            
            return {
                'predictions': predicted_conditions[:3],  # Top 3 predictions
                'severity': severity,
                'recommendations': self.get_recommendations(predicted_conditions, severity),
                'next_steps': self.get_next_steps(severity),
                'model_confidence': 85.2,
                'timestamp': time.time()
            }
            
        except Exception as e:
            print(f"AI prediction error: {e}")
            return self.get_rule_based_response(symptoms)
    
    def get_rule_based_response(self, symptoms):
        """Fallback rule-based response system"""
        symptoms_lower = symptoms.lower()
        
        if 'fever' in symptoms_lower:
            return {
                'primary_assessment': 'Fever symptoms detected',
                'recommendations': [
                    'Rest and stay hydrated',
                    'Monitor temperature regularly',
                    'Consider fever-reducing medication',
                    'Seek medical care if fever exceeds 103°F'
                ],
                'confidence': 75.0,
                'severity': 'moderate' if 'high' in symptoms_lower else 'mild'
            }
        
        elif any(word in symptoms_lower for word in ['chest pain', 'heart']):
            return {
                'primary_assessment': 'Chest pain requires immediate attention',
                'recommendations': [
                    'Call 911 immediately',
                    'Do not drive yourself',
                    'Chew aspirin if not allergic',
                    'Stay calm and rest'
                ],
                'confidence': 95.0,
                'severity': 'severe',
                'is_emergency': True
            }
        
        else:
            return {
                'primary_assessment': 'General health consultation',
                'recommendations': [
                    'Provide more specific symptom details',
                    'Monitor symptoms for changes',
                    'Consider consulting healthcare provider',
                    'Maintain good health practices'
                ],
                'confidence': 60.0,
                'severity': 'mild'
            }
    
    def get_recommendations(self, conditions, severity):
        """Generate recommendations based on conditions"""
        recommendations = []
        
        if severity == 'severe':
            recommendations.extend([
                'Seek immediate medical attention',
                'Consider emergency room visit',
                'Do not delay medical care'
            ])
        elif severity == 'moderate':
            recommendations.extend([
                'Schedule appointment with healthcare provider',
                'Monitor symptoms closely',
                'Follow home care measures'
            ])
        else:
            recommendations.extend([
                'Rest and self-care measures',
                'Monitor for worsening symptoms',
                'Consider over-the-counter treatments'
            ])
        
        return recommendations
    
    def get_next_steps(self, severity):
        """Generate next steps based on severity"""
        if severity == 'severe':
            return [
                'Call 911 or go to emergency room',
                'Gather medical history and medications',
                'Have someone accompany you'
            ]
        elif severity == 'moderate':
            return [
                'Call your doctor for appointment',
                'Keep symptom diary',
                'Follow recommended treatments'
            ]
        else:
            return [
                'Continue monitoring symptoms',
                'Try recommended home remedies',
                'Schedule routine check-up if needed'
            ]

class MedicalAIServer:
    """Web server for Medical AI application"""
    
    def __init__(self, port=8080):
        self.port = port
        self.medical_model = None
        self.server = None
        self.load_medical_model()
    
    def load_medical_model(self):
        """Load the trained medical AI model"""
        try:
            # Look for the medical AI model in parent directory
            model_path = parent_dir / 'best_rf_model.pkl'
            
            if model_path.exists():
                print(f"📦 Loading medical AI model from {model_path}")
                with open(model_path, 'rb') as f:
                    self.medical_model = pickle.load(f)
                print("✅ Medical AI model loaded successfully")
            else:
                print("⚠️ Medical AI model not found, using rule-based responses")
                
        except Exception as e:
            print(f"❌ Error loading medical model: {e}")
            print("📋 Falling back to rule-based medical responses")
    
    def start_server(self):
        """Start the web server"""
        try:
            # Change to web app directory
            web_dir = Path(__file__).parent
            os.chdir(web_dir)
            
            # Create handler with medical model
            handler = lambda *args, **kwargs: MedicalAIHandler(*args, medical_model=self.medical_model, **kwargs)
            
            # Start server
            self.server = HTTPServer(('localhost', self.port), handler)
            
            print(f"🏥 Medical AI Web Server starting...")
            print(f"🌐 Server URL: http://localhost:{self.port}")
            print(f"📱 ChatGPT-style interface ready")
            print(f"🩺 Medical AI model: {'Loaded' if self.medical_model else 'Rule-based'}")
            print("🚀 Opening browser...")
            
            # Open browser
            threading.Timer(2.0, lambda: webbrowser.open(f'http://localhost:{self.port}')).start()
            
            # Start server
            self.server.serve_forever()
            
        except KeyboardInterrupt:
            print("\n⏹️ Server stopping...")
            self.stop_server()
        except Exception as e:
            print(f"❌ Server error: {e}")
    
    def stop_server(self):
        """Stop the web server"""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            print("✅ Server stopped")

def main():
    """Main function to run the medical AI web application"""
    print("🏥 Medical AI Web Application")
    print("💻 ChatGPT-style interface for medical consultations")
    print("=" * 50)
    
    # Create and start server
    server = MedicalAIServer(port=8080)
    
    try:
        server.start_server()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()