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
import socket

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
        elif self.path == '/api/followup':
            self.handle_follow_up_check()
        else:
            self.send_error(404, "API endpoint not found")
    
    def handle_symptom_analysis(self):
        """Process symptom analysis request"""
        try:
            print(f"🔍 API /analyze called")
            # Parse request data
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length).decode('utf-8')
            data = json.loads(post_data)
            
            symptoms = data.get('symptoms', '')
            patient_info = data.get('patient_info', {})
            
            print(f"📝 Analyzing symptoms: {symptoms}")
            
            # Generate ChatGPT-style natural language response
            natural_response = self.generate_chatgpt_style_response(symptoms, patient_info)
            print(f"🤖 ChatGPT-style response generated")
            
            # Send JSON response with natural language
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            response = json.dumps({"response": natural_response})
            self.wfile.write(response.encode('utf-8'))
            
        except Exception as e:
            print(f"❌ Error in symptom analysis: {e}")
            import traceback
            traceback.print_exc()
            self.send_error(500, f"Analysis error: {str(e)}")
    
    def handle_follow_up_check(self):
        """Handle follow-up monitoring requests"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length).decode('utf-8')
            data = json.loads(post_data)
            
            patient_id = data.get('patient_id', 'anonymous')
            days_since_last = data.get('days_since_last', 0)
            current_symptoms = data.get('current_symptoms', '')
            previous_diagnosis = data.get('previous_diagnosis', '')
            
            follow_up_response = self.generate_follow_up_response(
                patient_id, days_since_last, current_symptoms, previous_diagnosis
            )
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            response = json.dumps({"response": follow_up_response})
            self.wfile.write(response.encode('utf-8'))
            
        except Exception as e:
            print(f"❌ Error in follow-up check: {e}")
            self.send_error(500, f"Follow-up error: {str(e)}")
    
    def generate_follow_up_response(self, patient_id, days_since_last, current_symptoms, previous_diagnosis):
        """Generate follow-up monitoring response"""
        response = f"📅 **Follow-up Check - Day {days_since_last}**\n\n"
        response += f"Thank you for checking back in! Let me assess how you're doing since our last consultation.\n\n"
        
        if 'diabetes' in previous_diagnosis.lower():
            response += "🩺 **Diabetes Monitoring Update:**\n\n"
            
            if current_symptoms:
                symptoms_lower = current_symptoms.lower()
                if any(word in symptoms_lower for word in ['better', 'improved', 'less']):
                    response += "✅ **Positive Progress:** It sounds like your symptoms may be improving. This is encouraging!\n\n"
                    response += "**Continue monitoring:**\n• Keep tracking symptoms daily\n• Have you been able to get blood work done?\n• Any dietary changes you've made?\n\n"
                elif any(word in symptoms_lower for word in ['worse', 'more', 'increased']):
                    response += "⚠️ **Symptom Progression:** Your symptoms appear to be worsening. This requires more urgent attention.\n\n"
                    response += "**URGENT ACTIONS:**\n• Contact your doctor immediately if not done already\n• Consider urgent care if doctor unavailable\n• Monitor for emergency symptoms (severe dehydration, vomiting)\n\n"
                else:
                    response += "📊 **Status Assessment:** Let me analyze your current symptom status.\n\n"
                    
                # Re-analyze current symptoms
                diabetes_assessment = self.assess_diabetes_likelihood(symptoms_lower, [])
                if diabetes_assessment['likelihood'] > 70:
                    response += f"The diabetes likelihood remains HIGH ({diabetes_assessment['likelihood']}%). "
                    response += "Getting medical testing is still critical.\n\n"
                
            else:
                response += "Please describe your current symptoms so I can assess any changes since our last check.\n\n"
            
            # Check if they've gotten tested
            response += "**Key follow-up questions:**\n"
            response += "• Have you been able to get blood glucose testing done?\n"
            response += "• Did you see a healthcare provider?\n"
            response += "• Are your symptoms the same, better, or worse?\n"
            response += "• Any new symptoms developed?\n\n"
            
            # Next check-in
            response += "**Next monitoring:** Please check back in 2-3 days or immediately if symptoms worsen significantly.\n\n"
            
        else:
            response += f"🩺 **Medical Follow-up for {previous_diagnosis}:**\n\n"
            response += "How are your symptoms progressing? Please describe any changes since our last consultation.\n\n"
            
        response += "**Remember:** This follow-up monitoring helps track your condition, but it doesn't replace professional medical care. If you're concerned about your symptoms or they're worsening, please contact your healthcare provider directly."
        
        return response
    
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
    
    def generate_chatgpt_style_response(self, symptoms, patient_info):
        """Generate natural language response like ChatGPT with medical predictions"""
        try:
            symptoms_lower = symptoms.lower()
            
            # Enhanced symptom pattern recognition
            detected_symptoms = self.extract_symptoms_from_text(symptoms_lower)
            
            # Advanced disease prediction with symptom clustering
            predicted_conditions = self.analyze_symptom_patterns(detected_symptoms, symptoms_lower)
            
            # Check for specific diabetes patterns
            diabetes_assessment = self.assess_diabetes_likelihood(symptoms_lower, detected_symptoms)
            
            if diabetes_assessment['likelihood'] > 70:
                # High diabetes likelihood - provide specific diabetes prediction
                return self.generate_diabetes_prediction_response(symptoms, diabetes_assessment, detected_symptoms)
            elif predicted_conditions:
                # Other medical conditions
                return self.generate_condition_prediction_response(symptoms, predicted_conditions, detected_symptoms)
            else:
                return self.generate_general_medical_response(symptoms)
            
        except Exception as e:
            print(f"Error generating ChatGPT response: {e}")
            return self.generate_general_medical_response(symptoms)

    def assess_diabetes_likelihood(self, symptoms_lower, detected_symptoms):
        """Assess likelihood of diabetes based on symptom patterns"""
        diabetes_score = 0
        diabetes_indicators = []
        
        symptom_names = [s['symptom'] for s in detected_symptoms] if detected_symptoms else []
        
        # Classic diabetes triad (3 P's)
        if 'increased_urination' in symptom_names or any(word in symptoms_lower for word in ['pee a lot', 'urinate more', 'frequent urination']):
            diabetes_score += 30
            diabetes_indicators.append('Polyuria (frequent urination)')
            
        if 'increased_thirst' in symptom_names or any(word in symptoms_lower for word in ['thirsty', 'dry mouth']):
            diabetes_score += 25
            diabetes_indicators.append('Polydipsia (excessive thirst)')
            
        if 'increased_hunger' in symptom_names or any(word in symptoms_lower for word in ['hungry all the time', 'always hungry', 'eating more']):
            diabetes_score += 25
            diabetes_indicators.append('Polyphagia (increased hunger)')
            
        # Supporting symptoms
        if 'dizziness' in symptom_names or 'dizzy' in symptoms_lower:
            diabetes_score += 15
            diabetes_indicators.append('Dizziness (blood sugar fluctuation)')
            
        if 'fatigue' in symptom_names or any(word in symptoms_lower for word in ['tired', 'weak', 'no energy']):
            diabetes_score += 10
            diabetes_indicators.append('Fatigue')
            
        if 'blurred_vision' in symptom_names or 'blurry vision' in symptoms_lower:
            diabetes_score += 15
            diabetes_indicators.append('Blurred vision')
            
        if 'weight_loss' in symptom_names or 'losing weight' in symptoms_lower:
            diabetes_score += 20
            diabetes_indicators.append('Unexplained weight loss')
        
        return {
            'likelihood': min(diabetes_score, 95),
            'indicators': diabetes_indicators,
            'classic_triad': diabetes_score >= 70,  # Has multiple classic symptoms
            'requires_testing': diabetes_score >= 40
        }

    def generate_diabetes_prediction_response(self, symptoms, diabetes_assessment, detected_symptoms):
        """Generate specific diabetes prediction response"""
        likelihood = diabetes_assessment['likelihood']
        
        response = f"Based on your symptoms - {symptoms.lower()} - I'm analyzing this through a medical AI lens.\n\n"
        
        # Main diabetes prediction
        if likelihood >= 85:
            response += f"🩺 **HIGH LIKELIHOOD: Type 2 Diabetes ({likelihood}% confidence)**\n\n"
            response += "Your symptom pattern strongly matches classic diabetes presentation. The combination you're describing is what we call the 'diabetes triad' in medicine.\n\n"
        elif likelihood >= 70:
            response += f"🩺 **PROBABLE: Type 2 Diabetes ({likelihood}% confidence)**\n\n"
            response += "Your symptoms are highly suggestive of diabetes or pre-diabetes. This warrants immediate medical evaluation.\n\n"
        else:
            response += f"🩺 **POSSIBLE: Diabetes-related condition ({likelihood}% confidence)**\n\n"
            response += "Your symptoms could indicate early diabetes or blood sugar issues that need investigation.\n\n"
        
        # Explain the indicators
        response += "**Diabetes indicators I've identified:**\n"
        for indicator in diabetes_assessment['indicators']:
            response += f"✓ {indicator}\n"
        response += "\n"
        
        # Medical explanation
        if diabetes_assessment['classic_triad']:
            response += "**Why this suggests diabetes:**\nThe 'three P's' (polyuria, polydipsia, polyphagia) you're experiencing are the hallmark signs of diabetes. When blood sugar is high, your kidneys work overtime to filter it out, causing frequent urination. This leads to dehydration and thirst. Meanwhile, your cells can't properly use the glucose, so you feel hungry despite eating.\n\n"
        
        # Immediate action plan
        response += "**IMMEDIATE ACTION PLAN:**\n"
        response += "🔬 **Get tested ASAP** - Request HbA1c and fasting glucose tests\n"
        response += "📞 **Call your doctor today** - Don't wait for a regular appointment\n"
        response += "📊 **Monitor symptoms** - Track urination frequency, thirst, hunger levels\n"
        response += "🚨 **Seek ER if severe** - Extreme thirst, vomiting, or breathing changes\n\n"
        
        # Follow-up monitoring
        response += "**FOLLOW-UP MONITORING:**\n"
        response += "I recommend we check in on your symptoms every 2-3 days until you get medical testing. "
        response += "Please return here to report:\n"
        response += "• Any worsening of symptoms\n"
        response += "• Test results when available\n"
        response += "• New symptoms that develop\n\n"
        
        # Lifestyle guidance
        response += "**INTERIM MANAGEMENT:**\n"
        response += "• Avoid sugary foods and drinks\n"
        response += "• Stay hydrated with water\n"
        response += "• Don't skip meals\n"
        response += "• Monitor for warning signs\n\n"
        
        response += "**Medical Disclaimer:** This AI assessment is based on symptom analysis and medical literature. Diabetes diagnosis requires laboratory testing and physician evaluation. If symptoms worsen or you feel unwell, seek immediate medical care."
        
        return response

    def generate_condition_prediction_response(self, symptoms, predicted_conditions, detected_symptoms):
        """Generate response for non-diabetes medical conditions"""
        predicted_conditions.sort(key=lambda x: x['confidence'], reverse=True)
        top_condition = predicted_conditions[0]
        
        response = f"Based on your symptoms - {symptoms.lower()} - here's my medical analysis:\n\n"
        
        # Main assessment
        if top_condition['confidence'] > 75:
            response += f"🩺 **LIKELY DIAGNOSIS: {top_condition['condition']} ({top_condition['confidence']:.0f}% confidence)**\n\n"
        elif top_condition['confidence'] > 60:
            response += f"🩺 **PROBABLE: {top_condition['condition']} ({top_condition['confidence']:.0f}% confidence)**\n\n"
        else:
            response += f"🩺 **POSSIBLE: {top_condition['condition']} ({top_condition['confidence']:.0f}% confidence)**\n\n"
        
        # Explain the symptoms
        if detected_symptoms:
            response += "**Symptoms identified:**\n"
            for symptom in detected_symptoms[:4]:
                response += f"✓ {symptom['symptom'].replace('_', ' ').title()}\n"
            response += "\n"
        
        # Medical explanation
        response += self.get_condition_explanation(top_condition['condition'])
        
        # Action plan based on severity
        response += "\n**RECOMMENDED ACTION PLAN:**\n"
        if top_condition['confidence'] > 75:
            response += f"📞 **Schedule medical appointment** - High confidence suggests {top_condition['condition']}\n"
            response += "🔬 **Likely tests needed** - Your doctor may want to confirm this diagnosis\n"
        else:
            response += "📞 **Consider medical consultation** - Multiple conditions possible\n"
            response += "📊 **Monitor symptoms** - Track changes and progression\n"
        
        response += "\n**FOLLOW-UP MONITORING:**\n"
        response += "Check back in 3-4 days to report symptom changes or if new symptoms develop.\n\n"
        
        # Alternative possibilities
        if len(predicted_conditions) > 1:
            response += "**OTHER POSSIBILITIES:**\n"
            for condition in predicted_conditions[1:3]:
                response += f"• {condition['condition']} ({condition['confidence']:.0f}% likelihood)\n"
            response += "\n"
        
        response += "**Medical Disclaimer:** This assessment is for informational purposes. Professional medical evaluation is recommended for proper diagnosis and treatment."
        
        return response

    def get_ai_prediction(self, symptoms, patient_info):
        """Advanced medical AI with natural language pattern recognition"""
        try:
            symptoms_lower = symptoms.lower()
            
            # Enhanced symptom pattern recognition
            detected_symptoms = self.extract_symptoms_from_text(symptoms_lower)
            
            # Advanced disease prediction with symptom clustering
            predicted_conditions = self.analyze_symptom_patterns(detected_symptoms, symptoms_lower)
            
            # Determine overall severity
            severity = self.assess_severity(detected_symptoms, symptoms_lower)
            
            # Sort predictions by confidence and medical urgency
            predicted_conditions.sort(key=lambda x: (x['urgency_score'], x['confidence']), reverse=True)
            
            return {
                'predictions': predicted_conditions[:3],
                'detected_symptoms': detected_symptoms,
                'severity': severity,
                'recommendations': self.get_recommendations(predicted_conditions, severity),
                'next_steps': self.get_next_steps(severity),
                'model_confidence': self.calculate_overall_confidence(predicted_conditions),
                'timestamp': time.time(),
                'alerts': self.check_urgent_conditions(detected_symptoms, symptoms_lower)
            }
        except Exception as e:
            return self.get_fallback_response(symptoms)
    
    def get_model_predictions(self, detected_symptoms, text):
        """Use the sklearn logistic model for predictions"""
        try:
            import numpy as np
            
            # Create input vector for symptoms
            if not hasattr(self, 'symptom_cols') or not self.symptom_cols:
                return []
                
            user_vector = np.zeros((1, len(self.symptom_cols)))
            
            # Map detected symptoms to model symptom columns
            symptom_names = [s['symptom'] for s in detected_symptoms]
            matched_symptoms = []
            
            for idx, symptom_col in enumerate(self.symptom_cols):
                # Convert model symptom column names to match our detection
                symptom_col_clean = symptom_col.replace('_', ' ').lower()
                
                for detected_symptom in symptom_names:
                    if (detected_symptom.replace('_', ' ').lower() in symptom_col_clean or 
                        symptom_col_clean in detected_symptom.replace('_', ' ').lower()):
                        user_vector[0, idx] = 1
                        matched_symptoms.append(symptom_col)
                        break
            
            # Make prediction using logistic regression
            if hasattr(self.medical_model, 'predict_proba'):
                # Get probability predictions
                predictions = self.medical_model.predict_proba(user_vector)[0]
                top_indices = np.argsort(predictions)[::-1][:3]
            else:
                # Fallback to regular prediction
                prediction = self.medical_model.predict(user_vector)[0]
                # Create a simple probability array
                predictions = np.zeros(len(self.disease_names))
                if prediction < len(self.disease_names):
                    predictions[prediction] = 0.9
                top_indices = [prediction]
            
            conditions = []
            for rank, idx in enumerate(top_indices):
                if idx < len(self.disease_names) and predictions[idx] > 0.01:  # Only include meaningful predictions
                    confidence = float(predictions[idx] * 100)
                    conditions.append({
                        'condition': self.disease_names[idx],
                        'confidence': confidence,
                        'severity': 'high' if confidence > 70 else ('moderate' if confidence > 40 else 'low'),
                        'urgency_score': confidence,
                        'matched_symptoms': matched_symptoms,
                        'description': f'AI-predicted condition based on symptom analysis',
                        'recommendation': f'Consult healthcare provider for proper diagnosis (AI confidence: {confidence:.1f}%)'
                    })
            
            return conditions
            
        except Exception as e:
            print(f"Model prediction error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def extract_symptoms_from_text(self, text):
        """Extract and categorize symptoms from natural language"""
        symptom_patterns = {
            # Diabetes symptoms (Classical triad + others)
            'increased_hunger': ['hungry a lot', 'hungry more', 'eating more', 'increased appetite', 'always hungry', 'starving', 'polyphagia'],
            'increased_urination': ['pee a lot', 'urinate more', 'bathroom more', 'frequent urination', 'polyuria', 'pee more'],
            'increased_thirst': ['thirsty a lot', 'drink more', 'dry mouth', 'polydipsia', 'dehydrated'],
            'dizziness': ['dizzy', 'lightheaded', 'faint', 'wooziness', 'vertigo', 'spinning'],
            'fatigue': ['tired', 'exhausted', 'weak', 'no energy', 'sleepy', 'lethargic'],
            'weight_loss': ['losing weight', 'weight loss', 'getting thinner', 'clothes loose'],
            'blurred_vision': ['blurry vision', 'can\'t see clearly', 'vision problems', 'eye problems'],
            
            # Cardiovascular symptoms
            'chest_pain': ['chest pain', 'chest pressure', 'chest tightness', 'heart pain'],
            'shortness_of_breath': ['short of breath', 'can\'t breathe', 'breathing problems', 'dyspnea'],
            'palpitations': ['heart racing', 'heart pounding', 'irregular heartbeat', 'palpitations'],
            
            # Respiratory symptoms
            'cough': ['coughing', 'cough', 'hacking', 'persistent cough'],
            'fever': ['fever', 'hot', 'temperature', 'feverish', 'burning up'],
            'sore_throat': ['sore throat', 'throat pain', 'scratchy throat'],
            
            # Gastrointestinal symptoms
            'nausea': ['nauseous', 'sick to stomach', 'queasy', 'want to vomit'],
            'vomiting': ['throwing up', 'vomiting', 'puking'],
            'diarrhea': ['loose stools', 'diarrhea', 'frequent bowel movements'],
            'abdominal_pain': ['stomach pain', 'belly pain', 'abdominal pain'],
            
            # Neurological symptoms
            'headache': ['headache', 'head pain', 'migraine', 'head hurts'],
            'confusion': ['confused', 'can\'t think', 'mental fog', 'disoriented'],
            'numbness': ['numb', 'tingling', 'pins and needles', 'no feeling'],
        }
        
        detected = []
        for symptom, patterns in symptom_patterns.items():
            for pattern in patterns:
                if pattern in text:
                    detected.append({
                        'symptom': symptom,
                        'pattern_matched': pattern,
                        'severity': self.estimate_symptom_severity(text, pattern)
                    })
                    break  # Only match once per symptom
        
        return detected
    
    def analyze_symptom_patterns(self, detected_symptoms, text):
        """Analyze symptom patterns to predict medical conditions"""
        conditions = []
        
        # Try to use the actual AI model first
        if hasattr(self, 'medical_model') and self.medical_model is not None:
            try:
                conditions = self.get_model_predictions(detected_symptoms, text)
                if conditions:  # If model gave predictions, return them
                    return conditions
            except Exception as e:
                print(f"Error using AI model: {e}")
                import traceback
                traceback.print_exc()
        
        # Fallback to rule-based analysis
        # Extract symptom names for easier checking
        symptom_names = [s['symptom'] for s in detected_symptoms]
        
        # DIABETES TYPE 2 - Classic triad + additional symptoms
        diabetes_score = 0
        diabetes_symptoms = []
        
        if 'increased_hunger' in symptom_names:
            diabetes_score += 25
            diabetes_symptoms.append('Polyphagia (increased hunger)')
        if 'increased_urination' in symptom_names:
            diabetes_score += 25
            diabetes_symptoms.append('Polyuria (frequent urination)')
        if 'increased_thirst' in symptom_names:
            diabetes_score += 25
            diabetes_symptoms.append('Polydipsia (excessive thirst)')
        if 'dizziness' in symptom_names:
            diabetes_score += 15
            diabetes_symptoms.append('Dizziness')
        if 'fatigue' in symptom_names:
            diabetes_score += 10
            diabetes_symptoms.append('Fatigue')
        if 'weight_loss' in symptom_names:
            diabetes_score += 15
            diabetes_symptoms.append('Unexplained weight loss')
        if 'blurred_vision' in symptom_names:
            diabetes_score += 15
            diabetes_symptoms.append('Blurred vision')
        
        if diabetes_score >= 40:  # High threshold for diabetes suggestion
            conditions.append({
                'condition': 'Type 2 Diabetes Mellitus',
                'confidence': min(diabetes_score, 95),
                'severity': 'moderate' if diabetes_score < 70 else 'high',
                'urgency_score': 85,
                'matched_symptoms': diabetes_symptoms,
                'description': 'A metabolic disorder characterized by high blood sugar levels',
                'recommendation': 'Blood glucose test and HbA1c recommended immediately'
            })
        
        # HYPOGLYCEMIA (Low blood sugar)
        hypo_score = 0
        hypo_symptoms = []
        
        if 'dizziness' in symptom_names and 'increased_hunger' in symptom_names:
            hypo_score += 40
            hypo_symptoms.extend(['Dizziness', 'Hunger'])
        if 'confusion' in symptom_names:
            hypo_score += 30
            hypo_symptoms.append('Confusion')
        if 'fatigue' in symptom_names:
            hypo_score += 15
            hypo_symptoms.append('Fatigue')
        
        if hypo_score >= 35:
            conditions.append({
                'condition': 'Hypoglycemia (Low Blood Sugar)',
                'confidence': min(hypo_score, 90),
                'severity': 'moderate',
                'urgency_score': 75,
                'matched_symptoms': hypo_symptoms,
                'description': 'Blood glucose levels below normal range',
                'recommendation': 'Check blood sugar immediately, consume fast-acting carbohydrates'
            })
        
        # CARDIOVASCULAR CONDITIONS
        cardio_score = 0
        cardio_symptoms = []
        
        if 'chest_pain' in symptom_names:
            cardio_score += 50
            cardio_symptoms.append('Chest pain')
        if 'shortness_of_breath' in symptom_names:
            cardio_score += 30
            cardio_symptoms.append('Shortness of breath')
        if 'dizziness' in symptom_names:
            cardio_score += 20
            cardio_symptoms.append('Dizziness')
        if 'palpitations' in symptom_names:
            cardio_score += 25
            cardio_symptoms.append('Palpitations')
        
        if cardio_score >= 50:
            conditions.append({
                'condition': 'Cardiovascular Condition',
                'confidence': min(cardio_score, 95),
                'severity': 'high',
                'urgency_score': 95,
                'matched_symptoms': cardio_symptoms,
                'description': 'Potential heart or circulation problem',
                'recommendation': 'Seek immediate medical attention - call 911 if severe'
            })
        
        # RESPIRATORY INFECTIONS
        resp_score = 0
        resp_symptoms = []
        
        if 'cough' in symptom_names:
            resp_score += 30
            resp_symptoms.append('Cough')
        if 'fever' in symptom_names:
            resp_score += 35
            resp_symptoms.append('Fever')
        if 'sore_throat' in symptom_names:
            resp_score += 25
            resp_symptoms.append('Sore throat')
        if 'fatigue' in symptom_names:
            resp_score += 15
            resp_symptoms.append('Fatigue')
        
        if resp_score >= 45:
            conditions.append({
                'condition': 'Upper Respiratory Infection',
                'confidence': min(resp_score, 85),
                'severity': 'mild' if resp_score < 60 else 'moderate',
                'urgency_score': 30,
                'matched_symptoms': resp_symptoms,
                'description': 'Viral or bacterial respiratory infection',
                'recommendation': 'Rest, fluids, consider seeing healthcare provider if symptoms worsen'
            })
        
        # GASTROINTESTINAL CONDITIONS
        gi_score = 0
        gi_symptoms = []
        
        if 'nausea' in symptom_names:
            gi_score += 25
            gi_symptoms.append('Nausea')
        if 'vomiting' in symptom_names:
            gi_score += 35
            gi_symptoms.append('Vomiting')
        if 'diarrhea' in symptom_names:
            gi_score += 30
            gi_symptoms.append('Diarrhea')
        if 'abdominal_pain' in symptom_names:
            gi_score += 25
            gi_symptoms.append('Abdominal pain')
        
        if gi_score >= 40:
            conditions.append({
                'condition': 'Gastroenteritis',
                'confidence': min(gi_score, 80),
                'severity': 'mild' if gi_score < 60 else 'moderate',
                'urgency_score': 40,
                'matched_symptoms': gi_symptoms,
                'description': 'Inflammation of stomach and intestines',
                'recommendation': 'Stay hydrated, bland diet, seek care if severe or persistent'
            })
        
        # If no specific patterns detected, provide general assessment
        if not conditions and detected_symptoms:
            conditions.append({
                'condition': 'General Symptom Assessment',
                'confidence': 60,
                'severity': 'mild',
                'urgency_score': 25,
                'matched_symptoms': [s['symptom'].replace('_', ' ').title() for s in detected_symptoms],
                'description': 'Multiple symptoms detected requiring evaluation',
                'recommendation': 'Consider consulting healthcare provider for proper assessment'
            })
        
        return conditions
    
    def check_urgent_conditions(self, detected_symptoms, text):
        """Check for urgent medical conditions requiring immediate attention"""
        alerts = []
        symptom_names = [s['symptom'] for s in detected_symptoms]
        
        # Diabetes emergency (DKA risk)
        diabetes_urgent = ['increased_hunger', 'increased_urination', 'increased_thirst']
        if len([s for s in diabetes_urgent if s in symptom_names]) >= 2:
            if 'dizziness' in symptom_names or 'confusion' in symptom_names:
                alerts.append({
                    'type': 'urgent',
                    'condition': 'Possible Diabetic Emergency',
                    'message': 'The combination of excessive hunger, urination, and dizziness may indicate diabetes or diabetic complications. Seek medical attention promptly.',
                    'action': 'Get blood glucose tested immediately'
                })
        
        # Cardiovascular emergency
        if 'chest_pain' in symptom_names:
            alerts.append({
                'type': 'emergency',
                'condition': 'Possible Cardiac Event',
                'message': 'Chest pain requires immediate evaluation to rule out heart attack.',
                'action': 'Call 911 or go to emergency room immediately'
            })
        
        return alerts
    
    def estimate_symptom_severity(self, text, pattern):
        """Estimate severity of a symptom based on context"""
        severity_indicators = {
            'mild': ['a little', 'slightly', 'minor', 'mild'],
            'moderate': ['moderate', 'noticeable', 'quite a bit'],
            'severe': ['a lot', 'very', 'extremely', 'severe', 'terrible', 'unbearable']
        }
        
        for severity, indicators in severity_indicators.items():
            if any(indicator in text for indicator in indicators):
                return severity
        return 'moderate'  # default
    
    def assess_severity(self, detected_symptoms, text):
        """Assess overall severity based on symptoms and context"""
        severity_scores = {'mild': 1, 'moderate': 2, 'severe': 3}
        
        if not detected_symptoms:
            return 'mild'
        
        # Check for severe indicators in text
        severe_indicators = ['emergency', 'urgent', 'can\'t', 'unable', 'severe', 'terrible']
        if any(indicator in text for indicator in severe_indicators):
            return 'severe'
        
        # Calculate average severity
        avg_severity = sum(severity_scores.get(s.get('severity', 'moderate'), 2) for s in detected_symptoms) / len(detected_symptoms)
        
        if avg_severity >= 2.5:
            return 'severe'
        elif avg_severity >= 1.5:
            return 'moderate'
        else:
            return 'mild'
    
    def calculate_overall_confidence(self, predictions):
        """Calculate overall confidence based on predictions"""
        if not predictions:
            return 50.0
        
        # Weight by urgency and confidence
        weighted_confidence = sum(p['confidence'] * (p['urgency_score'] / 100) for p in predictions)
        return min(weighted_confidence / len(predictions), 95.0)
    
    def get_condition_explanation(self, condition):
        """Provide natural language explanation of medical conditions"""
        explanations = {
            'Diabetes': "**Medical Analysis:** Type 2 diabetes occurs when your body becomes resistant to insulin or doesn't produce enough insulin to maintain normal glucose levels. The classic triad you're experiencing happens because:\n• **Frequent urination (polyuria)**: High blood sugar overwhelms the kidneys\n• **Excessive thirst (polydipsia)**: Body tries to replace lost fluids\n• **Increased hunger (polyphagia)**: Cells can't access glucose properly\n\nThis is a serious but manageable condition that requires medical treatment.",
            
            'Hypoglycemia': "**Medical Analysis:** Low blood sugar episodes can cause the symptoms you're describing. Your brain relies on glucose for energy, so when levels drop, you experience dizziness, hunger, and weakness. This needs evaluation to determine the underlying cause.",
            
            'Dehydration': "**Medical Analysis:** Dehydration affects multiple body systems. When fluid levels drop, blood pressure can decrease (causing dizziness), and your body may trigger hunger signals as it seeks water from food sources. Severe dehydration requires medical attention.",
            
            'Urinary Tract Infection': "**Medical Analysis:** UTIs cause inflammation in the urinary system, leading to frequent, urgent urination. Left untreated, infections can spread to the kidneys. This is highly treatable with appropriate antibiotics after proper diagnosis.",
            
            'Hyperthyroidism': "**Medical Analysis:** An overactive thyroid accelerates metabolism, which can cause increased appetite, frequent urination, weight loss despite eating more, and cardiovascular symptoms like dizziness. This hormonal condition requires medical management.",
            
            'Anxiety Disorder': "**Medical Analysis:** Anxiety activates the sympathetic nervous system, which can cause physical symptoms including dizziness, changes in appetite, and frequent urination. The mind-body connection in anxiety is well-documented medically.",
            
            'Pre-diabetes': "**Medical Analysis:** Pre-diabetes means blood sugar levels are higher than normal but not yet in the diabetic range. You may experience mild versions of diabetes symptoms. This is a critical time for intervention to prevent progression to full diabetes.",
            
            'Metabolic Syndrome': "**Medical Analysis:** A cluster of conditions including insulin resistance, which can cause diabetes-like symptoms. This represents increased risk for diabetes and cardiovascular disease, requiring lifestyle and possibly medical intervention."
        }
        
        explanation = explanations.get(condition, f"**Medical Analysis:** {condition} is a medical condition that correlates with your symptom pattern. Professional evaluation will provide specific diagnostic clarity and treatment options.")
        return f"{explanation}\n"

    def generate_general_medical_response(self, symptoms):
        """Generate a general medical response when specific diagnosis is unclear"""
        return f"""I understand you're experiencing some concerning symptoms: {symptoms.lower()}

While I can't provide a definitive diagnosis without a proper medical examination, I can offer some general guidance:

**What these symptoms might indicate:**
The combination of symptoms you've described could be related to several different conditions. Some possibilities include metabolic changes, hormonal imbalances, infections, or stress-related factors.

**What I recommend:**
• **Schedule an appointment with your healthcare provider** - they can perform proper tests and examinations
• **Keep a symptom diary** - note when symptoms occur, their severity, and any potential triggers
• **Stay hydrated and maintain regular meals** - this can help with dizziness and some other symptoms
• **Monitor for any worsening** - seek immediate care if symptoms become severe

**When to seek immediate care:**
• If you experience severe dizziness or fainting
• If symptoms suddenly worsen significantly  
• If you develop additional concerning symptoms

**Important:** This is general health information and not a substitute for professional medical advice. Your healthcare provider can properly evaluate your symptoms and recommend appropriate tests or treatments based on your individual health history."""

    def get_fallback_response(self, symptoms):
        """Fallback response when advanced analysis fails"""
        return {
            'predictions': [{
                'condition': 'Symptom Assessment Needed',
                'confidence': 60,
                'severity': 'moderate',
                'urgency_score': 50,
                'matched_symptoms': ['Multiple symptoms reported'],
                'description': 'Professional medical evaluation recommended',
                'recommendation': 'Please consult with a healthcare provider for proper diagnosis'
            }],
            'detected_symptoms': [],
            'severity': 'moderate',
            'recommendations': ['Consult healthcare provider'],
            'next_steps': ['Schedule appointment with doctor'],
            'model_confidence': 60.0,
            'timestamp': time.time(),
            'alerts': []
        }
    
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
    
    def get_local_ip(self):
        """Get local IP address for mobile access"""
        try:
            # Connect to a remote server to determine local IP
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(('8.8.8.8', 80))
                local_ip = s.getsockname()[0]
            return local_ip
        except:
            return '192.168.1.x'  # Fallback message
    
    def load_medical_model(self):
        """Load the trained medical AI model"""
        try:
            # Try logistic model first (lighter and faster)
            logistic_path = Path('logistic_model.pkl')
            
            if logistic_path.exists():
                print(f"📦 Loading medical AI model from {logistic_path}")
                import pickle
                with open(logistic_path, 'rb') as f:
                    self.medical_model = pickle.load(f)
                print("✅ Medical AI model loaded successfully")
                
                # Load training data to get symptom columns and disease names
                training_path = Path('Training.csv')
                if training_path.exists():
                    import pandas as pd
                    df = pd.read_csv(training_path)
                    # Get symptom columns (exclude the target column)
                    target_col = 'prognosis'
                    self.symptom_cols = [col for col in df.columns if col != target_col]
                    self.disease_names = sorted(df[target_col].unique())
                    print(f"✅ Loaded {len(self.symptom_cols)} symptoms and {len(self.disease_names)} diseases")
                else:
                    print("⚠️ Training.csv not found, using basic symptoms")
                    self.symptom_cols = []
                    self.disease_names = []
            else:
                print("⚠️ Medical AI model not found, using rule-based responses")
                
        except Exception as e:
            print(f"❌ Error loading medical model: {e}")
            import traceback
            traceback.print_exc()
            print("📋 Falling back to rule-based medical responses")
    
    def start_server(self):
        """Start the web server"""
        try:
            # Change to web app directory
            web_dir = Path(__file__).parent
            os.chdir(web_dir)
            
            # Create handler with medical model
            handler = lambda *args, **kwargs: MedicalAIHandler(*args, medical_model=self.medical_model, **kwargs)
            
            # Start server (allow connections from any device on network)
            self.server = HTTPServer(('0.0.0.0', self.port), handler)
            
            print(f"🏥 Medical AI Web Server starting...")
            print(f"🌐 Local URL: http://localhost:{self.port}")
            print(f"📱 Mobile URL: http://{self.get_local_ip()}:{self.port}")
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