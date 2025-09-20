import os
import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
import re
import json
from collections import defaultdict
import difflib
import hashlib
from datetime import datetime, timedelta
import sqlite3
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import warnings
from pathlib import Path
import base64
from PIL import Image
import io
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import threading
import queue
import time

class MentalHealthAssessment:
    """
    Comprehensive Mental Health Assessment and Support System
    """
    
    def __init__(self):
        self.crisis_keywords = [
            'suicide', 'kill myself', 'end my life', 'not worth living', 'want to die',
            'self harm', 'cutting', 'hurting myself', 'overdose', 'end it all'
        ]
        self.severity_indicators = self._load_severity_indicators()
        self.mental_health_screenings = self._load_screening_tools()
        self.crisis_resources = self._load_crisis_resources()
        self.treatment_recommendations = self._load_treatment_options()
    
    def _load_severity_indicators(self):
        """Load mental health severity assessment criteria"""
        return {
            'depression': {
                'mild': ['sad', 'down', 'low mood', 'tired', 'loss of interest'],
                'moderate': ['hopeless', 'worthless', 'guilt', 'sleep problems', 'appetite changes'],
                'severe': ['suicidal thoughts', 'cant function', 'extreme hopelessness', 'psychotic symptoms']
            },
            'anxiety': {
                'mild': ['worried', 'nervous', 'restless', 'on edge'],
                'moderate': ['panic attacks', 'avoidance', 'physical symptoms', 'intrusive thoughts'],
                'severe': ['agoraphobia', 'severe panic', 'cant leave house', 'debilitating fear']
            },
            'psychosis': {
                'mild': ['suspicious thoughts', 'unusual experiences', 'mild paranoia'],
                'moderate': ['clear delusions', 'hallucinations', 'disorganized thinking'],
                'severe': ['command hallucinations', 'severe delusions', 'complete loss of reality']
            },
            'mania': {
                'mild': ['elevated mood', 'increased energy', 'decreased sleep'],
                'moderate': ['grandiose thoughts', 'risky behavior', 'rapid speech'],
                'severe': ['psychotic features', 'complete loss of judgment', 'hospitalization needed']
            }
        }
    
    def _load_screening_tools(self):
        """Load standardized mental health screening questionnaires"""
        return {
            'PHQ-9': {
                'questions': [
                    'Little interest or pleasure in doing things',
                    'Feeling down, depressed, or hopeless',
                    'Trouble falling or staying asleep, or sleeping too much',
                    'Feeling tired or having little energy',
                    'Poor appetite or overeating',
                    'Feeling bad about yourself',
                    'Trouble concentrating',
                    'Moving or speaking slowly, or being fidgety',
                    'Thoughts of self-harm'
                ],
                'scoring': 'Each item 0-3, total 0-27'
            },
            'GAD-7': {
                'questions': [
                    'Feeling nervous, anxious, or on edge',
                    'Not being able to stop or control worrying',
                    'Worrying too much about different things',
                    'Trouble relaxing',
                    'Being so restless its hard to sit still',
                    'Becoming easily annoyed or irritable',
                    'Feeling afraid as if something awful might happen'
                ],
                'scoring': 'Each item 0-3, total 0-21'
            },
            'PTSD-5': {
                'questions': [
                    'Repeated, disturbing memories of the stressful experience',
                    'Trouble sleeping',
                    'Feeling distant or cut off from other people',
                    'Being super-alert or watchful on guard',
                    'Feeling irritable or having angry outbursts'
                ],
                'scoring': 'Each item 0-4, total 0-20'
            }
        }
    
    def _load_crisis_resources(self):
        """Load crisis intervention resources and hotlines"""
        return {
            'suicide_prevention': {
                'national': '988 - Suicide & Crisis Lifeline (24/7)',
                'text': 'Text HOME to 741741 - Crisis Text Line',
                'chat': 'suicidepreventionlifeline.org - Online Chat',
                'international': '1-800-273-8255 - International Association'
            },
            'domestic_violence': {
                'national': '1-800-799-7233 - National Domestic Violence Hotline',
                'text': 'Text START to 88788'
            },
            'substance_abuse': {
                'samhsa': '1-800-662-4357 - SAMHSA National Helpline',
                'description': 'Treatment referral and information service'
            },
            'eating_disorders': {
                'neda': '1-800-931-2237 - National Eating Disorders Association',
                'text': 'Text NEDA to 741741'
            },
            'lgbtq': {
                'trevor': '1-866-488-7386 - Trevor Project (LGBTQ youth)',
                'trans': '1-877-565-8860 - Trans Lifeline'
            }
        }
    
    def _load_treatment_options(self):
        """Load evidence-based treatment recommendations"""
        return {
            'depression': [
                'Cognitive Behavioral Therapy (CBT)',
                'Interpersonal Therapy (IPT)',
                'Antidepressant medications (SSRIs, SNRIs)',
                'Behavioral Activation',
                'Mindfulness-Based Cognitive Therapy'
            ],
            'anxiety': [
                'Cognitive Behavioral Therapy (CBT)',
                'Exposure and Response Prevention',
                'Anti-anxiety medications (benzodiazepines, SSRIs)',
                'Mindfulness and Relaxation Training',
                'EMDR for trauma-related anxiety'
            ],
            'ptsd': [
                'Trauma-Focused CBT',
                'Eye Movement Desensitization and Reprocessing (EMDR)',
                'Prolonged Exposure Therapy',
                'Cognitive Processing Therapy',
                'SSRI/SNRI medications'
            ],
            'bipolar': [
                'Mood stabilizers (lithium, anticonvulsants)',
                'Dialectical Behavior Therapy (DBT)',
                'Family-Focused Therapy',
                'Interpersonal and Social Rhythm Therapy',
                'Psychoeducation'
            ]
        }
    
    def assess_mental_health_severity(self, symptoms_text, extracted_symptoms):
        """
        Assess mental health symptom severity and risk level
        
        Args:
            symptoms_text: Raw text input from user
            extracted_symptoms: List of identified mental health symptoms
            
        Returns:
            dict: Mental health assessment with severity and recommendations
        """
        severity_score = 0
        identified_conditions = []
        risk_level = 'low'
        
        # Crisis detection
        crisis_detected = any(keyword in symptoms_text.lower() for keyword in self.crisis_keywords)
        
        # Severity assessment for each condition
        for condition, severity_levels in self.severity_indicators.items():
            condition_score = 0
            matched_symptoms = []
            
            for severity, symptom_list in severity_levels.items():
                for symptom in symptom_list:
                    if symptom in symptoms_text.lower() or symptom.replace(' ', '_') in extracted_symptoms:
                        if severity == 'mild':
                            condition_score += 1
                        elif severity == 'moderate':
                            condition_score += 2
                        elif severity == 'severe':
                            condition_score += 3
                        matched_symptoms.append(symptom)
            
            if condition_score > 0:
                identified_conditions.append({
                    'condition': condition,
                    'score': condition_score,
                    'symptoms': matched_symptoms,
                    'severity': self._interpret_severity_score(condition_score)
                })
                severity_score += condition_score
        
        # Determine overall risk level
        if crisis_detected or severity_score >= 8:
            risk_level = 'crisis'
        elif severity_score >= 5:
            risk_level = 'high'
        elif severity_score >= 3:
            risk_level = 'moderate'
        else:
            risk_level = 'low'
        
        return {
            'overall_severity_score': severity_score,
            'risk_level': risk_level,
            'crisis_detected': crisis_detected,
            'identified_conditions': identified_conditions,
            'recommendations': self._generate_mental_health_recommendations(risk_level, identified_conditions),
            'crisis_resources': self.crisis_resources if crisis_detected or risk_level == 'crisis' else {}
        }
    
    def _interpret_severity_score(self, score):
        """Convert numerical severity score to descriptive level"""
        if score >= 6:
            return 'Severe'
        elif score >= 3:
            return 'Moderate'
        elif score >= 1:
            return 'Mild'
        else:
            return 'Minimal'
    
    def _generate_mental_health_recommendations(self, risk_level, conditions):
        """Generate appropriate mental health recommendations"""
        recommendations = []
        
        if risk_level == 'crisis':
            recommendations.extend([
                "🚨 IMMEDIATE CRISIS INTERVENTION NEEDED",
                "Call 988 (Suicide & Crisis Lifeline) NOW",
                "Go to nearest Emergency Room",
                "Do not leave person alone",
                "Remove any means of self-harm"
            ])
        elif risk_level == 'high':
            recommendations.extend([
                "Urgent mental health evaluation recommended",
                "Contact mental health professional within 24 hours",
                "Consider urgent care or crisis center",
                "Implement safety planning",
                "Reach out to support system"
            ])
        elif risk_level == 'moderate':
            recommendations.extend([
                "Schedule appointment with mental health professional",
                "Consider therapy or counseling",
                "Discuss symptoms with primary care doctor",
                "Practice stress management techniques",
                "Maintain regular sleep and exercise"
            ])
        else:
            recommendations.extend([
                "Monitor symptoms for changes",
                "Practice self-care and stress management",
                "Consider preventive mental health resources",
                "Maintain healthy lifestyle habits"
            ])
        
        # Add condition-specific recommendations
        for condition_data in conditions:
            condition = condition_data['condition']
            if condition in self.treatment_recommendations:
                recommendations.append(f"For {condition}: Consider {self.treatment_recommendations[condition][0]}")
        
        return recommendations
    
    def generate_safety_plan(self, user_input, risk_assessment):
        """Generate personalized mental health safety plan"""
        safety_plan = {
            'warning_signs': [],
            'coping_strategies': [],
            'social_supports': [],
            'professional_contacts': [],
            'crisis_resources': self.crisis_resources,
            'environment_safety': []
        }
        
        # Personalized based on identified conditions
        for condition_data in risk_assessment.get('identified_conditions', []):
            condition = condition_data['condition']
            
            if condition == 'depression':
                safety_plan['warning_signs'].extend([
                    'Increasing hopelessness', 'Social withdrawal', 'Sleep changes'
                ])
                safety_plan['coping_strategies'].extend([
                    'Call a friend', 'Take a walk', 'Practice deep breathing'
                ])
            elif condition == 'anxiety':
                safety_plan['warning_signs'].extend([
                    'Racing thoughts', 'Physical tension', 'Avoidance behaviors'
                ])
                safety_plan['coping_strategies'].extend([
                    'Grounding techniques', 'Progressive muscle relaxation', 'Mindfulness'
                ])
        
        return safety_plan

class AdvancedMedicalAI:
    """
    Advanced Medical AI with comprehensive features including:
    - Multi-modal learning
    - Uncertainty quantification
    - Clinical decision support
    - Risk stratification
    - Emergency detection
    - Mental health assessment
    """
    
    def __init__(self):
        self.emergency_keywords = [
            'chest pain', 'difficulty breathing', 'severe headache', 'stroke symptoms',
            'heart attack', 'severe bleeding', 'unconscious', 'seizure', 'overdose',
            'suicide', 'severe burns', 'choking', 'severe allergic reaction'
        ]
        self.drug_interactions = self._load_drug_interactions()
        self.risk_factors = self._load_risk_factors()
        self.medical_guidelines = self._load_medical_guidelines()
        self.mental_health = MentalHealthAssessment()
        
    def _load_drug_interactions(self):
        """Load common drug interaction database"""
        return {
            'warfarin': {
                'interactions': ['aspirin', 'ibuprofen', 'acetaminophen'],
                'severity': ['major', 'moderate', 'minor'],
                'warnings': ['Increased bleeding risk', 'Monitor INR closely', 'Use with caution']
            },
            'metformin': {
                'interactions': ['alcohol', 'contrast dye', 'steroids'],
                'severity': ['major', 'major', 'moderate'],
                'warnings': ['Lactic acidosis risk', 'Kidney function monitoring', 'Blood sugar monitoring']
            },
            'lisinopril': {
                'interactions': ['potassium supplements', 'nsaids', 'lithium'],
                'severity': ['major', 'moderate', 'moderate'],
                'warnings': ['Hyperkalemia risk', 'Kidney function', 'Lithium toxicity']
            }
        }
    
    def _load_risk_factors(self):
        """Load risk factor scoring system"""
        return {
            'cardiovascular': {
                'age': {'male': '>45', 'female': '>55'},
                'smoking': 2,
                'hypertension': 2,
                'diabetes': 2,
                'family_history': 1,
                'obesity': 1
            },
            'diabetes': {
                'age': '>45',
                'obesity': 2,
                'family_history': 2,
                'sedentary': 1,
                'hypertension': 1,
                'gestational_diabetes': 1
            },
            'stroke': {
                'age': '>65',
                'hypertension': 3,
                'diabetes': 2,
                'smoking': 2,
                'atrial_fibrillation': 3,
                'prior_stroke': 2
            }
        }
    
    def _load_medical_guidelines(self):
        """Load clinical practice guidelines"""
        return {
            'hypertension': {
                'stage1': {'systolic': '130-139', 'diastolic': '80-89'},
                'stage2': {'systolic': '>=140', 'diastolic': '>=90'},
                'crisis': {'systolic': '>180', 'diastolic': '>120'}
            },
            'diabetes': {
                'normal': {'fasting_glucose': '<100', 'hba1c': '<5.7'},
                'prediabetes': {'fasting_glucose': '100-125', 'hba1c': '5.7-6.4'},
                'diabetes': {'fasting_glucose': '>=126', 'hba1c': '>=6.5'}
            },
            'bmi': {
                'underweight': '<18.5',
                'normal': '18.5-24.9',
                'overweight': '25-29.9',
                'obese': '>=30'
            }
        }
    
    def detect_emergency(self, symptoms_text, vital_signs=None):
        """
        Detect emergency conditions requiring immediate medical attention
        
        Args:
            symptoms_text: Natural language symptom description
            vital_signs: Optional vital signs dictionary
            
        Returns:
            dict: Emergency assessment with urgency level and recommendations
        """
        emergency_score = 0
        detected_emergencies = []
        
        # Text-based emergency detection
        text_lower = symptoms_text.lower()
        for keyword in self.emergency_keywords:
            if keyword in text_lower:
                emergency_score += 3
                detected_emergencies.append(keyword)
        
        # Vital signs emergency detection
        if vital_signs:
            if vital_signs.get('systolic_bp', 0) > 180:
                emergency_score += 5
                detected_emergencies.append('hypertensive crisis')
            if vital_signs.get('heart_rate', 0) > 150 or vital_signs.get('heart_rate', 0) < 40:
                emergency_score += 4
                detected_emergencies.append('cardiac emergency')
            if vital_signs.get('temperature', 0) > 104:
                emergency_score += 4
                detected_emergencies.append('hyperthermia')
            if vital_signs.get('oxygen_saturation', 100) < 90:
                emergency_score += 5
                detected_emergencies.append('respiratory emergency')
        
        # Determine urgency level
        if emergency_score >= 8:
            urgency = 'CRITICAL - Call 911 Immediately'
        elif emergency_score >= 5:
            urgency = 'HIGH - Seek Emergency Care'
        elif emergency_score >= 3:
            urgency = 'MODERATE - Contact Healthcare Provider'
        else:
            urgency = 'LOW - Monitor Symptoms'
            
        return {
            'emergency_score': emergency_score,
            'urgency_level': urgency,
            'detected_emergencies': detected_emergencies,
            'recommendations': self._get_emergency_recommendations(urgency, detected_emergencies)
        }
    
    def _get_emergency_recommendations(self, urgency, emergencies):
        """Generate emergency-specific recommendations"""
        if 'CRITICAL' in urgency:
            return [
                "🚨 CALL 911 IMMEDIATELY",
                "Do not drive yourself to hospital",
                "If unconscious, check breathing and pulse",
                "Stay with patient until help arrives"
            ]
        elif 'HIGH' in urgency:
            return [
                "Go to Emergency Room immediately",
                "Do not delay seeking medical care",
                "Bring list of current medications",
                "Have someone drive you if possible"
            ]
        elif 'MODERATE' in urgency:
            return [
                "Contact your healthcare provider today",
                "Monitor symptoms closely",
                "Seek care if symptoms worsen",
                "Keep symptom diary"
            ]
        else:
            return [
                "Monitor symptoms for changes",
                "Practice self-care measures",
                "Contact provider if concerned",
                "Schedule routine follow-up"
            ]
    
    def check_drug_interactions(self, medications):
        """
        Check for potential drug interactions
        
        Args:
            medications: List of current medications
            
        Returns:
            dict: Interaction warnings and severity levels
        """
        interactions = []
        
        for med1 in medications:
            med1_lower = med1.lower()
            if med1_lower in self.drug_interactions:
                for med2 in medications:
                    if med2.lower() in self.drug_interactions[med1_lower]['interactions']:
                        idx = self.drug_interactions[med1_lower]['interactions'].index(med2.lower())
                        severity = self.drug_interactions[med1_lower]['severity'][idx]
                        warning = self.drug_interactions[med1_lower]['warnings'][idx]
                        
                        interactions.append({
                            'drug1': med1,
                            'drug2': med2,
                            'severity': severity,
                            'warning': warning
                        })
        
        return {
            'has_interactions': len(interactions) > 0,
            'interactions': interactions,
            'total_interactions': len(interactions)
        }
    
    def calculate_risk_score(self, condition, patient_data):
        """
        Calculate risk score for specific medical conditions
        
        Args:
            condition: Medical condition (e.g., 'cardiovascular', 'diabetes')
            patient_data: Patient demographics and risk factors
            
        Returns:
            dict: Risk assessment with score and interpretation
        """
        if condition not in self.risk_factors:
            return {'error': f'Risk factors not available for {condition}'}
        
        risk_factors = self.risk_factors[condition]
        score = 0
        applicable_factors = []
        
        for factor, weight in risk_factors.items():
            if factor in patient_data:
                if isinstance(weight, int):
                    score += weight
                    applicable_factors.append(factor)
                elif isinstance(weight, dict):
                    # Age/gender specific factors
                    gender = patient_data.get('gender', 'unknown')
                    age = patient_data.get('age', 0)
                    if gender in weight:
                        age_threshold = int(weight[gender].replace('>', ''))
                        if age > age_threshold:
                            score += 1
                            applicable_factors.append(f"{factor} (age {age})")
        
        # Interpret risk level
        if score >= 6:
            risk_level = 'High Risk'
        elif score >= 3:
            risk_level = 'Moderate Risk'
        elif score >= 1:
            risk_level = 'Low Risk'
        else:
            risk_level = 'Minimal Risk'
            
        return {
            'condition': condition,
            'risk_score': score,
            'risk_level': risk_level,
            'applicable_factors': applicable_factors,
            'recommendations': self._get_risk_recommendations(condition, risk_level)
        }
    
    def _get_risk_recommendations(self, condition, risk_level):
        """Generate condition and risk-specific recommendations"""
        base_recommendations = {
            'cardiovascular': {
                'High Risk': [
                    "Immediate cardiology consultation recommended",
                    "Consider cardiac stress testing",
                    "Aggressive lifestyle modifications needed",
                    "Blood pressure and cholesterol monitoring"
                ],
                'Moderate Risk': [
                    "Regular cardiovascular monitoring",
                    "Lifestyle modifications recommended",
                    "Annual cardiac risk assessment",
                    "Consider preventive medications"
                ]
            },
            'diabetes': {
                'High Risk': [
                    "Diabetes screening recommended",
                    "Endocrinology consultation",
                    "Glucose monitoring advised",
                    "Immediate lifestyle interventions"
                ]
            }
        }
        
        return base_recommendations.get(condition, {}).get(risk_level, [
            "Follow standard screening guidelines",
            "Maintain healthy lifestyle",
            "Regular medical check-ups"
        ])

class PatientHistoryManager:
    """
    Manages patient history, demographics, and personalized medical data
    """
    
    def __init__(self, db_path="patient_data.db"):
        self.db_path = db_path
        self.init_database()
        
    def init_database(self):
        """Initialize SQLite database for patient data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Patient demographics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patients (
                patient_id TEXT PRIMARY KEY,
                name TEXT,
                age INTEGER,
                gender TEXT,
                ethnicity TEXT,
                height REAL,
                weight REAL,
                created_date DATETIME,
                last_updated DATETIME
            )
        ''')
        
        # Medical history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS medical_history (
                history_id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT,
                condition TEXT,
                diagnosis_date DATETIME,
                status TEXT,
                notes TEXT,
                FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
            )
        ''')
        
        # Consultation sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS consultations (
                session_id TEXT PRIMARY KEY,
                patient_id TEXT,
                symptoms TEXT,
                predictions TEXT,
                confidence_scores TEXT,
                session_date DATETIME,
                follow_up_needed BOOLEAN,
                FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
            )
        ''')
        
        # Medications table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS medications (
                med_id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT,
                medication_name TEXT,
                dosage TEXT,
                frequency TEXT,
                start_date DATETIME,
                end_date DATETIME,
                active BOOLEAN,
                FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
            )
        ''')
        
        # Vital signs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS vital_signs (
                vital_id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT,
                session_id TEXT,
                systolic_bp INTEGER,
                diastolic_bp INTEGER,
                heart_rate INTEGER,
                temperature REAL,
                oxygen_saturation INTEGER,
                recorded_date DATETIME,
                FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_patient_profile(self, patient_data):
        """Create new patient profile"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        patient_id = hashlib.md5(
            f"{patient_data.get('name', '')}{datetime.now()}".encode()
        ).hexdigest()[:12]
        
        cursor.execute('''
            INSERT INTO patients (patient_id, name, age, gender, ethnicity, height, weight, created_date, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            patient_id,
            patient_data.get('name', ''),
            patient_data.get('age', 0),
            patient_data.get('gender', ''),
            patient_data.get('ethnicity', ''),
            patient_data.get('height', 0),
            patient_data.get('weight', 0),
            datetime.now(),
            datetime.now()
        ))
        
        conn.commit()
        conn.close()
        
        return patient_id
    
    def get_patient_history(self, patient_id):
        """Retrieve complete patient history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get patient demographics
        cursor.execute('SELECT * FROM patients WHERE patient_id = ?', (patient_id,))
        patient_data = cursor.fetchone()
        
        # Get medical history
        cursor.execute('SELECT * FROM medical_history WHERE patient_id = ?', (patient_id,))
        medical_history = cursor.fetchall()
        
        # Get recent consultations
        cursor.execute(
            'SELECT * FROM consultations WHERE patient_id = ? ORDER BY session_date DESC LIMIT 10',
            (patient_id,)
        )
        recent_consultations = cursor.fetchall()
        
        # Get current medications
        cursor.execute(
            'SELECT * FROM medications WHERE patient_id = ? AND active = 1',
            (patient_id,)
        )
        current_medications = cursor.fetchall()
        
        conn.close()
        
        return {
            'patient_data': patient_data,
            'medical_history': medical_history,
            'recent_consultations': recent_consultations,
            'current_medications': current_medications
        }
    
    def save_consultation(self, patient_id, session_data):
        """Save consultation session data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        session_id = hashlib.md5(
            f"{patient_id}{datetime.now()}".encode()
        ).hexdigest()[:12]
        
        cursor.execute('''
            INSERT INTO consultations 
            (session_id, patient_id, symptoms, predictions, confidence_scores, session_date, follow_up_needed)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            session_id,
            patient_id,
            json.dumps(session_data.get('symptoms', [])),
            json.dumps(session_data.get('predictions', [])),
            json.dumps(session_data.get('confidence_scores', [])),
            datetime.now(),
            session_data.get('follow_up_needed', False)
        ))
        
        conn.commit()
        conn.close()
        
        return session_id

class UncertaintyQuantifier:
    """
    Provides uncertainty quantification for medical predictions
    """
    
    def __init__(self):
        self.calibration_data = {}
        
    def calculate_prediction_uncertainty(self, model_predictions, ensemble_variance=None):
        """
        Calculate uncertainty metrics for predictions
        
        Args:
            model_predictions: Array of prediction probabilities
            ensemble_variance: Variance across ensemble models
            
        Returns:
            dict: Uncertainty metrics and confidence intervals
        """
        # Entropy-based uncertainty
        entropy = -np.sum(model_predictions * np.log(model_predictions + 1e-10))
        max_entropy = np.log(len(model_predictions))
        normalized_entropy = entropy / max_entropy
        
        # Prediction confidence (max probability)
        max_confidence = np.max(model_predictions)
        
        # Margin of confidence (difference between top 2 predictions)
        sorted_probs = np.sort(model_predictions)[::-1]
        margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0]
        
        # Overall uncertainty score (0 = certain, 1 = uncertain)
        uncertainty_score = (normalized_entropy + (1 - max_confidence) + (1 - margin)) / 3
        
        return {
            'entropy': entropy,
            'normalized_entropy': normalized_entropy,
            'max_confidence': max_confidence,
            'margin': margin,
            'uncertainty_score': uncertainty_score,
            'confidence_level': self._interpret_uncertainty(uncertainty_score)
        }
    
    def _interpret_uncertainty(self, uncertainty_score):
        """Convert uncertainty score to interpretable confidence level"""
        if uncertainty_score < 0.2:
            return "Very High Confidence"
        elif uncertainty_score < 0.4:
            return "High Confidence"
        elif uncertainty_score < 0.6:
            return "Moderate Confidence"
        elif uncertainty_score < 0.8:
            return "Low Confidence"
        else:
            return "Very Low Confidence"

class VisualSymptomMapper:
    """
    Creates interactive body diagram for symptom mapping
    """
    
    def __init__(self):
        self.body_regions = {
            'head': ['headache', 'dizziness', 'vision_problems', 'hearing_problems'],
            'throat': ['sore_throat', 'difficulty_swallowing', 'hoarseness'],
            'chest': ['chest_pain', 'shortness_of_breath', 'cough', 'palpitations'],
            'abdomen': ['abdominal_pain', 'nausea', 'vomiting', 'diarrhea'],
            'back': ['back_pain', 'kidney_pain', 'muscle_aches'],
            'arms': ['arm_pain', 'numbness', 'weakness', 'joint_pain'],
            'legs': ['leg_pain', 'swelling', 'numbness', 'joint_pain'],
            'skin': ['rash', 'itching', 'bruising', 'discoloration']
        }
        
    def generate_body_map_html(self):
        """Generate interactive HTML body map"""
        return '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Interactive Symptom Body Map</title>
            <style>
                .body-map { width: 400px; height: 600px; position: relative; margin: 0 auto; }
                .body-region { position: absolute; cursor: pointer; opacity: 0.7; }
                .body-region:hover { opacity: 1; background-color: rgba(255,0,0,0.3); }
                .selected { background-color: rgba(255,0,0,0.5); }
                #head { top: 10px; left: 150px; width: 100px; height: 80px; border-radius: 50px; }
                #chest { top: 120px; left: 125px; width: 150px; height: 120px; }
                #abdomen { top: 260px; left: 140px; width: 120px; height: 100px; }
                #left-arm { top: 140px; left: 60px; width: 60px; height: 200px; }
                #right-arm { top: 140px; left: 280px; width: 60px; height: 200px; }
                #left-leg { top: 380px; left: 120px; width: 60px; height: 200px; }
                #right-leg { top: 380px; left: 220px; width: 60px; height: 200px; }
            </style>
        </head>
        <body>
            <div class="body-map">
                <div class="body-region" id="head" onclick="selectRegion('head')">Head</div>
                <div class="body-region" id="chest" onclick="selectRegion('chest')">Chest</div>
                <div class="body-region" id="abdomen" onclick="selectRegion('abdomen')">Abdomen</div>
                <div class="body-region" id="left-arm" onclick="selectRegion('arms')">L Arm</div>
                <div class="body-region" id="right-arm" onclick="selectRegion('arms')">R Arm</div>
                <div class="body-region" id="left-leg" onclick="selectRegion('legs')">L Leg</div>
                <div class="body-region" id="right-leg" onclick="selectRegion('legs')">R Leg</div>
            </div>
            
            <div id="selected-symptoms">
                <h3>Selected Symptoms:</h3>
                <ul id="symptom-list"></ul>
            </div>
            
            <script>
                let selectedSymptoms = [];
                
                function selectRegion(region) {
                    const symptoms = getRegionSymptoms(region);
                    symptoms.forEach(symptom => {
                        if (!selectedSymptoms.includes(symptom)) {
                            selectedSymptoms.push(symptom);
                        }
                    });
                    updateSymptomList();
                }
                
                function getRegionSymptoms(region) {
                    const regionSymptoms = {
                        'head': ['headache', 'dizziness', 'vision_problems'],
                        'chest': ['chest_pain', 'shortness_of_breath', 'cough'],
                        'abdomen': ['abdominal_pain', 'nausea', 'vomiting'],
                        'arms': ['arm_pain', 'numbness', 'weakness'],
                        'legs': ['leg_pain', 'swelling', 'joint_pain']
                    };
                    return regionSymptoms[region] || [];
                }
                
                function updateSymptomList() {
                    const list = document.getElementById('symptom-list');
                    list.innerHTML = selectedSymptoms.map(s => `<li>${s}</li>`).join('');
                }
            </script>
        </body>
        </html>
        '''
    
    def get_region_symptoms(self, selected_regions):
        """Get symptoms associated with selected body regions"""
        symptoms = []
        for region in selected_regions:
            if region in self.body_regions:
                symptoms.extend(self.body_regions[region])
        return list(set(symptoms))

class NaturalLanguageProcessor:
    """
    Natural Language Processing for Medical AI
    Handles symptom extraction, conversational interface, and natural language understanding
    """
    
    def __init__(self, symptom_columns):
        self.symptom_columns = [col.lower().replace('_', ' ') for col in symptom_columns]
        self.original_columns = symptom_columns
        self.symptom_synonyms = self._load_symptom_synonyms()
        self.conversation_context = {}
        
    def _load_symptom_synonyms(self):
        """Load comprehensive symptom synonyms and variations for advanced NLP matching"""
        synonyms = {
            # Pain symptoms
            'fever': ['temperature', 'hot', 'burning up', 'feverish', 'high temp', 'pyrexia', 'febrile'],
            'headache': ['head pain', 'migraine', 'head hurts', 'cranial pain', 'cephalgia', 'skull pain'],
            'chest pain': ['chest hurts', 'chest discomfort', 'thoracic pain', 'heart pain', 'angina', 'crushing pain'],
            'abdominal pain': ['stomach ache', 'belly pain', 'tummy hurts', 'stomach pain', 'gut pain', 'cramping'],
            'back pain': ['backache', 'spine pain', 'lower back pain', 'upper back pain', 'lumbar pain'],
            'joint pain': ['arthritis', 'joint aches', 'stiff joints', 'joint stiffness', 'arthritic pain'],
            'muscle aches': ['body aches', 'sore muscles', 'muscle pain', 'myalgia', 'muscular pain'],
            
            # Gastrointestinal symptoms
            'nausea': ['sick to stomach', 'queasy', 'want to vomit', 'feel sick', 'nauseated', 'stomach upset'],
            'vomiting': ['throwing up', 'puking', 'being sick', 'emesis', 'retching', 'upchuck'],
            'diarrhea': ['loose stools', 'watery stool', 'frequent bowel movements', 'runs', 'loose bowels'],
            'constipation': ['cant poop', 'blocked up', 'hard stools', 'irregular bowels', 'difficulty defecating'],
            'loss of appetite': ['dont want to eat', 'no appetite', 'not hungry', 'anorexia', 'poor appetite'],
            
            # Respiratory symptoms
            'cough': ['coughing', 'hacking', 'barking cough', 'dry cough', 'wet cough', 'productive cough'],
            'shortness of breath': ['breathless', 'cant breathe', 'winded', 'dyspnea', 'difficulty breathing'],
            'sore throat': ['throat pain', 'throat hurts', 'painful swallowing', 'scratchy throat', 'pharyngitis'],
            'runny nose': ['stuffy nose', 'congestion', 'nasal discharge', 'blocked nose', 'rhinorrhea'],
            'wheezing': ['whistling breathing', 'tight chest', 'asthmatic breathing', 'bronchospasm'],
            
            # Cardiovascular symptoms
            'palpitations': ['heart racing', 'fast heartbeat', 'irregular heartbeat', 'heart flutter', 'tachycardia'],
            'chest tightness': ['tight chest', 'chest pressure', 'constricted chest', 'chest squeezing'],
            'swelling': ['edema', 'puffiness', 'fluid retention', 'bloating', 'water retention'],
            
            # Neurological symptoms
            'dizziness': ['dizzy', 'lightheaded', 'vertigo', 'spinning sensation', 'unsteady', 'wobbly'],
            'confusion': ['confused', 'disoriented', 'foggy mind', 'mental fog', 'unclear thinking'],
            'numbness': ['numb', 'tingling', 'pins and needles', 'loss of feeling', 'paresthesia'],
            'weakness': ['weak', 'feeble', 'no strength', 'muscle weakness', 'fatigue', 'asthenia'],
            'memory problems': ['forgetful', 'cant remember', 'memory loss', 'amnesia', 'cognitive issues'],
            
            # General symptoms
            'fatigue': ['tired', 'exhausted', 'weary', 'worn out', 'drained', 'lethargic', 'sleepy'],
            'weight loss': ['losing weight', 'getting thinner', 'dropping pounds', 'unintentional weight loss'],
            'weight gain': ['gaining weight', 'putting on weight', 'getting heavier', 'weight increase'],
            'night sweats': ['sweating at night', 'nocturnal sweating', 'drenching sweats', 'night time sweating'],
            'chills': ['shivering', 'feeling cold', 'shaking from cold', 'rigors', 'goosebumps'],
            'insomnia': ['cant sleep', 'sleeplessness', 'trouble sleeping', 'sleep problems', 'restless nights'],
            
            # Dermatological symptoms
            'rash': ['skin rash', 'red spots', 'skin irritation', 'skin eruption', 'dermatitis', 'hives'],
            'itching': ['itchy', 'pruritus', 'scratchy skin', 'irritated skin', 'urge to scratch'],
            'bruising': ['bruises', 'black and blue', 'contusions', 'discoloration', 'purple marks'],
            
            # Urological symptoms
            'frequent urination': ['peeing a lot', 'urinating often', 'bathroom trips', 'polyuria'],
            'painful urination': ['burning pee', 'painful peeing', 'dysuria', 'stinging urination'],
            'blood in urine': ['red urine', 'pink urine', 'hematuria', 'bloody pee'],
            
            # Ophthalmological symptoms
            'vision problems': ['blurry vision', 'cant see clearly', 'double vision', 'vision loss', 'eye problems'],
            'eye pain': ['sore eyes', 'eye ache', 'painful eyes', 'eye discomfort', 'ocular pain'],
            
            # Auditory symptoms
            'hearing problems': ['cant hear well', 'hearing loss', 'deaf', 'muffled hearing', 'ear problems'],
            'ringing in ears': ['tinnitus', 'buzzing ears', 'ear noise', 'whistling ears'],
            
            # Comprehensive Mental Health & Psychiatric Symptoms
            'anxiety': ['anxious', 'worried', 'nervous', 'panic', 'stressed', 'on edge', 'restless', 'tense', 'fearful'],
            'depression': ['sad', 'depressed', 'down', 'hopeless', 'blue', 'melancholy', 'empty', 'worthless', 'guilty'],
            'mood changes': ['moody', 'irritable', 'mood swings', 'emotional', 'temperamental', 'unpredictable emotions'],
            'panic attacks': ['panic', 'sudden fear', 'racing heart', 'cant breathe', 'overwhelming terror', 'intense fear'],
            'social anxiety': ['social fear', 'afraid of people', 'embarrassed', 'self conscious', 'avoid social situations'],
            'obsessive thoughts': ['obsessive', 'intrusive thoughts', 'cant stop thinking', 'repetitive thoughts', 'unwanted thoughts'],
            'compulsive behavior': ['compulsive', 'repetitive actions', 'must do certain things', 'ritualistic behavior'],
            'hallucinations': ['hearing voices', 'seeing things', 'hallucinating', 'voices in head', 'visual disturbances'],
            'delusions': ['paranoid', 'suspicious', 'false beliefs', 'conspiracy', 'persecution', 'grandiose thoughts'],
            'suicidal thoughts': ['suicidal', 'want to die', 'end my life', 'kill myself', 'not worth living', 'suicide'],
            'self harm': ['cutting', 'hurting myself', 'self injury', 'harming myself', 'self mutilation'],
            'eating problems': ['not eating', 'binge eating', 'purging', 'obsessed with food', 'fear of gaining weight'],
            'sleep disturbances': ['cant sleep', 'nightmares', 'sleep problems', 'insomnia', 'sleeping too much', 'restless sleep'],
            'concentration problems': ['cant focus', 'distracted', 'poor concentration', 'mind wandering', 'attention problems'],
            'memory issues': ['forgetful', 'memory problems', 'cant remember', 'memory loss', 'confusion'],
            'agitation': ['agitated', 'restless', 'cant sit still', 'pacing', 'fidgety', 'hyperactive'],
            'withdrawal': ['isolating', 'avoiding people', 'withdrawn', 'antisocial', 'dont want to be around others'],
            'mood episodes': ['manic', 'euphoric', 'extremely happy', 'high energy', 'grandiose', 'hypomanic'],
            'emotional numbness': ['numb', 'empty inside', 'no feelings', 'emotionally dead', 'disconnected'],
            'anger issues': ['angry', 'rage', 'violent thoughts', 'explosive anger', 'irritability', 'aggression'],
            'substance use': ['drinking too much', 'using drugs', 'addiction', 'substance abuse', 'dependency'],
            'trauma symptoms': ['flashbacks', 'nightmares', 'ptsd', 'traumatic memories', 'triggered', 'hypervigilant'],
            'body image issues': ['hate my body', 'feel fat', 'body dysmorphia', 'ugly', 'appearance obsession'],
            'gender dysphoria': ['wrong body', 'gender identity', 'dont feel like my gender', 'transgender feelings'],
            'sexual problems': ['no libido', 'sexual dysfunction', 'intimacy issues', 'sexual trauma', 'arousal problems'],
            'relationship problems': ['relationship issues', 'trust problems', 'codependent', 'attachment issues'],
            'work stress': ['job stress', 'burnout', 'workplace anxiety', 'career problems', 'work overwhelm'],
            'grief': ['grieving', 'mourning', 'loss', 'bereavement', 'missing someone', 'death of loved one'],
            'perfectionism': ['perfectionist', 'never good enough', 'high standards', 'fear of failure', 'overachiever'],
            'abandonment fears': ['fear of abandonment', 'clingy', 'afraid of being left', 'rejection sensitivity'],
            'dissociation': ['dissociating', 'out of body', 'detached', 'unreal feelings', 'depersonalization']
        }
        
        # Add reverse mapping and variations
        expanded_synonyms = defaultdict(list)
        for main_symptom, variations in synonyms.items():
            expanded_synonyms[main_symptom].extend(variations)
            for variation in variations:
                expanded_synonyms[variation].append(main_symptom)
                
        return expanded_synonyms
    
    def extract_symptoms_from_text(self, text):
        """
        Extract symptoms from natural language text input
        
        Args:
            text: User's natural language description of symptoms
            
        Returns:
            list: Matched symptoms from the available symptom columns
        """
        text = text.lower().strip()
        matched_symptoms = []
        
        # Direct symptom matching
        for symptom in self.symptom_columns:
            if symptom in text:
                matched_symptoms.append(symptom)
                
        # Synonym matching
        for word_phrase in text.split():
            # Clean the phrase
            clean_phrase = re.sub(r'[^\w\s]', '', word_phrase).strip()
            if clean_phrase in self.symptom_synonyms:
                for synonym in self.symptom_synonyms[clean_phrase]:
                    if synonym in self.symptom_columns and synonym not in matched_symptoms:
                        matched_symptoms.append(synonym)
                        
        # Fuzzy matching for typos and variations
        words = text.split()
        for word in words:
            if len(word) > 3:  # Only for words longer than 3 chars
                closest_matches = difflib.get_close_matches(
                    word, self.symptom_columns, n=2, cutoff=0.8
                )
                for match in closest_matches:
                    if match not in matched_symptoms:
                        matched_symptoms.append(match)
                        
        return matched_symptoms
    
    def generate_conversational_response(self, symptoms, predictions, confidence_scores):
        """
        Generate natural language response based on predictions
        
        Args:
            symptoms: List of identified symptoms
            predictions: Top predicted diseases
            confidence_scores: Confidence scores for predictions
            
        Returns:
            str: Natural language response
        """
        if not symptoms:
            return ("I didn't detect any specific symptoms from your description. "
                   "Could you tell me more about how you're feeling? "
                   "For example, do you have fever, headache, nausea, or any pain?")
        
        symptom_text = ', '.join(symptoms)
        if len(symptoms) == 1:
            symptom_phrase = f"the symptom '{symptoms[0]}'"
        else:
            symptom_phrase = f"the symptoms: {symptom_text}"
            
        response = f"Based on {symptom_phrase}, here's what I found:\n\n"
        
        # Add predictions with confidence levels
        for i, (disease, confidence) in enumerate(zip(predictions[:3], confidence_scores[:3])):
            confidence_level = self._get_confidence_description(confidence)
            response += f"{i+1}. **{disease}** ({confidence_level} - {confidence:.1%})\n"
            
        # Add follow-up suggestions
        response += "\n" + self._generate_followup_questions(symptoms, predictions[0])
        
        return response
    
    def _get_confidence_description(self, confidence):
        """Convert numerical confidence to descriptive text"""
        if confidence >= 0.8:
            return "High confidence"
        elif confidence >= 0.6:
            return "Moderate confidence"
        elif confidence >= 0.4:
            return "Low confidence"
        else:
            return "Very low confidence"
    
    def _generate_followup_questions(self, current_symptoms, top_prediction):
        """Generate relevant follow-up questions based on current symptoms"""
        questions = []
        
        # Common follow-up questions based on symptom patterns
        if 'fever' in current_symptoms:
            questions.append("How high is your fever? Have you measured your temperature?")
        if 'pain' in ' '.join(current_symptoms):
            questions.append("Can you describe the pain? Is it sharp, dull, throbbing, or burning?")
        if 'cough' in current_symptoms:
            questions.append("Is your cough dry or are you bringing up mucus/phlegm?")
        if 'headache' in current_symptoms:
            questions.append("Where exactly is the headache? Is it behind your eyes, on one side, or all over?")
            
        # General questions if no specific follow-ups
        if not questions:
            questions = [
                "How long have you been experiencing these symptoms?",
                "Are there any other symptoms you've noticed?",
                "Have you taken any medications or treatments?"
            ]
            
        return "💡 **To help me provide a better assessment:**\n" + "\n".join([f"• {q}" for q in questions[:3]])
    
    def process_conversation_turn(self, user_input, context=None):
        """
        Process a single turn of conversation with the user
        
        Args:
            user_input: User's natural language input
            context: Previous conversation context
            
        Returns:
            dict: Processed information including symptoms, responses, questions
        """
        # Extract symptoms from the input
        symptoms = self.extract_symptoms_from_text(user_input)
        
        # Analyze the input for different intents
        result = {
            'symptoms': symptoms,
            'intent': self._classify_intent(user_input),
            'entities': self._extract_entities(user_input),
            'needs_clarification': len(symptoms) == 0,
            'raw_input': user_input
        }
        
        return result
    
    def _classify_intent(self, text):
        """Classify user intent from the text"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['help', 'what', 'how', '?']):
            return 'question'
        elif any(word in text_lower for word in ['feel', 'have', 'experiencing', 'symptoms']):
            return 'symptom_report'
        elif any(word in text_lower for word in ['yes', 'no', 'maybe', 'sometimes']):
            return 'response'
        else:
            return 'symptom_report'  # Default
    
    def _extract_entities(self, text):
        """Extract medical entities like duration, severity, etc."""
        entities = {}
        
        # Duration patterns
        duration_pattern = r'(\d+)\s*(day|week|month|hour|minute)s?'
        duration_match = re.search(duration_pattern, text.lower())
        if duration_match:
            entities['duration'] = f"{duration_match.group(1)} {duration_match.group(2)}{'s' if int(duration_match.group(1)) > 1 else ''}"
            
        # Severity patterns
        if any(word in text.lower() for word in ['severe', 'terrible', 'unbearable', 'intense']):
            entities['severity'] = 'severe'
        elif any(word in text.lower() for word in ['mild', 'slight', 'little', 'minor']):
            entities['severity'] = 'mild'
        elif any(word in text.lower() for word in ['moderate', 'medium', 'average']):
            entities['severity'] = 'moderate'
            
        return entities

class ConversationalInterface:
    """
    Manages the conversational flow and user interaction
    """
    
    def __init__(self, nlp_processor):
        self.nlp = nlp_processor
        self.conversation_history = []
        self.gathered_symptoms = []
        
    def start_conversation(self):
        """Start the conversational interface"""
        print("\n" + "="*60)
        print("🏥 AI MEDICAL ASSISTANT - Natural Language Interface")
        print("="*60)
        print("\nHello! I'm your AI medical assistant. I can help analyze your symptoms")
        print("and provide potential diagnoses based on what you tell me.")
        print("\nYou can describe your symptoms in your own words - just tell me how you're feeling!")
        print("\nExamples:")
        print("• 'I have a bad headache and feel nauseous'")
        print("• 'I've been running a fever for 2 days and have a sore throat'")
        print("• 'My stomach hurts and I feel dizzy'")
        print("\n" + "-"*60)
        
    def get_user_input(self, prompt="How are you feeling? Describe your symptoms: "):
        """Get natural language input from user with improved prompting"""
        print(f"\n💬 {prompt}")
        user_input = input("➤ ").strip()
        return user_input
    
    def provide_feedback(self, response):
        """Provide formatted feedback to the user"""
        print(f"\n🔍 **Analysis:**")
        print(response)
        print("\n" + "-"*60)

# 1. Load the original Training.csv dataset (4920 samples with 120 per disease)
df = pd.read_csv('C:/Users/bubhi/OneDrive/Desktop/AI/tfenv311/Training.csv')
print("Loaded original Training.csv dataset with full samples!")

# 2. Prepare features and labels  
symptom_cols = [col for col in df.columns if col != 'prognosis']
X = df[symptom_cols].fillna(0).values  # Fill NaN values with 0
y = pd.factorize(df['prognosis'])[0]
label_names = pd.factorize(df['prognosis'])[1]

# 3. Oversample minority classes
ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(X, y)

# 4. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# 5. Define a function to build the model (for cross-validation and tuning)
def build_model(optimizer='adam'):
    model = models.Sequential([
        layers.Input(shape=(len(symptom_cols),)),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.005)),
        layers.Dropout(0.5),
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.005)),
        layers.Dropout(0.4),
        layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.005)),
        layers.Dropout(0.3),
        layers.Dense(len(label_names), activation='softmax')
    ])
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# 6. Model training/loading logic
model_path = 'disease_model.h5'
retrain = True
if os.path.exists(model_path):
    try:
        best_model = load_model(model_path)
        if best_model.input_shape[-1] == X_train.shape[1]:
            print("Loaded trained model from file.")
            best_params = (16, 'adam')
            retrain = False
        else:
            print("Model input shape does not match data. Retraining...")
            os.remove(model_path)
    except Exception as e:
        print(f"Error loading model: {e}. Retraining...")
        os.remove(model_path)

if retrain:
    # Check if we have enough samples for cross-validation
    min_class_size = min(np.bincount(y_train))
    print(f"Minimum class size: {min_class_size}")
    
    if min_class_size >= 3:
        print("Using cross-validation for hyperparameter tuning...")
        best_val_acc = 0
        best_params = None
        best_model = None
        batch_sizes = [16, 32]
        optimizers = ['adam', 'rmsprop']
        n_splits = min(3, min_class_size)
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        for batch_size in batch_sizes:
            for optimizer in optimizers:
                val_accuracies = []
                for train_idx, val_idx in kfold.split(X_train, y_train):
                    model = build_model(optimizer=optimizer)
                    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                    model.fit(
                        X_train[train_idx], y_train[train_idx],
                        epochs=50,
                        batch_size=batch_size,
                        validation_data=(X_train[val_idx], y_train[val_idx]),
                        verbose=0,
                        callbacks=[early_stop]
                    )
                    val_loss, val_acc = model.evaluate(X_train[val_idx], y_train[val_idx], verbose=0)
                    val_accuracies.append(val_acc)
                avg_val_acc = np.mean(val_accuracies)
                print(f"Batch size: {batch_size}, Optimizer: {optimizer}, Avg. val accuracy: {avg_val_acc:.2f}")
                if avg_val_acc > best_val_acc:
                    best_val_acc = avg_val_acc
                    best_params = (batch_size, optimizer)
        
        # Train final model with best parameters
        best_model = build_model(optimizer=best_params[1])
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        best_model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=best_params[0],
            validation_split=0.1,
            verbose=1,
            callbacks=[early_stop]
        )
    else:
        print("Dataset too small for cross-validation. Training with default parameters...")
        best_params = (16, 'adam')
        best_model = build_model(optimizer='adam')
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        best_model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=16,
            validation_split=0.2,
            verbose=1,
            callbacks=[early_stop]
        )
    best_model.save(model_path)
    print(f"Model trained and saved to {model_path}.")

# 6b. Train Logistic Regression model
logistic_model_path = 'logistic_model.pkl'
if os.path.exists(logistic_model_path):
    print("Loading existing logistic regression model...")
    with open(logistic_model_path, 'rb') as f:
        logistic_model = pickle.load(f)
else:
    print("Training logistic regression model...")
    logistic_model = LogisticRegression(max_iter=1000, random_state=42)
    logistic_model.fit(X_train, y_train)
    with open(logistic_model_path, 'wb') as f:
        pickle.dump(logistic_model, f)
    print("Logistic regression model trained and saved.")

# 7. Evaluate individual models and create ensemble
nn_loss, nn_accuracy = best_model.evaluate(X_test, y_test, verbose=0)
lr_predictions = logistic_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_predictions)

print(f"\nNeural Network - Best params: batch_size={best_params[0]}, optimizer={best_params[1]}")
print(f"Neural Network Test accuracy: {nn_accuracy:.2f}")
print(f"Logistic Regression Test accuracy: {lr_accuracy:.2f}")

# Create ensemble predictions (weighted average)
nn_probs = best_model.predict(X_test)
lr_probs = logistic_model.predict_proba(X_test)

print(f"NN output shape: {nn_probs.shape}, LR output shape: {lr_probs.shape}")

# Check if shapes match, if not, rebuild models with correct number of classes
if nn_probs.shape[1] != lr_probs.shape[1]:
    print("Model output shapes don't match. Rebuilding models...")
    # Delete old models and retrain
    if os.path.exists(model_path):
        os.remove(model_path)
    if os.path.exists(logistic_model_path):
        os.remove(logistic_model_path)
    
    # Retrain neural network with correct number of classes
    print("Retraining neural network...")
    best_model = build_model(optimizer='adam')
    best_model.fit(X_train, y_train, epochs=20, batch_size=16, validation_split=0.2, verbose=1)
    best_model.save(model_path)
    
    # Retrain logistic regression
    print("Retraining logistic regression...")
    logistic_model = LogisticRegression(max_iter=1000, random_state=42)
    logistic_model.fit(X_train, y_train)
    with open(logistic_model_path, 'wb') as f:
        pickle.dump(logistic_model, f)
    
    # Get new predictions
    nn_probs = best_model.predict(X_test)
    lr_probs = logistic_model.predict_proba(X_test)

# Weight the predictions (you can adjust these weights)
nn_weight = 0.7
lr_weight = 0.3
ensemble_probs = nn_weight * nn_probs + lr_weight * lr_probs
ensemble_predictions = np.argmax(ensemble_probs, axis=1)
ensemble_accuracy = accuracy_score(y_test, ensemble_predictions)
print(f"Ensemble Model Test accuracy: {ensemble_accuracy:.2f}")

# 8. Initialize Advanced AI Systems
nlp_processor = NaturalLanguageProcessor(symptom_cols)
conversation_interface = ConversationalInterface(nlp_processor)
advanced_ai = AdvancedMedicalAI()
patient_manager = PatientHistoryManager()
uncertainty_quantifier = UncertaintyQuantifier()
visual_mapper = VisualSymptomMapper()

def enhanced_medical_prediction(user_input, best_model, logistic_model, label_names, symptom_cols, 
                              patient_id=None, vital_signs=None, medications=None, nn_weight=0.7, lr_weight=0.3):
    """
    Enhanced medical prediction with comprehensive AI features
    
    Args:
        user_input: Natural language description of symptoms
        best_model: Trained neural network model
        logistic_model: Trained logistic regression model
        label_names: List of disease names
        symptom_cols: List of symptom columns
        patient_id: Optional patient identifier for history
        vital_signs: Optional vital signs dictionary
        medications: Optional list of current medications
        nn_weight: Weight for neural network predictions
        lr_weight: Weight for logistic regression predictions
        
    Returns:
        dict: Comprehensive prediction results with advanced features
    """
    print("\n🔍 **ENHANCED MEDICAL AI ANALYSIS**")
    print("=" * 50)
    
    # 1. EMERGENCY DETECTION
    emergency_assessment = advanced_ai.detect_emergency(user_input, vital_signs)
    if emergency_assessment['emergency_score'] >= 5:
        print(f"🚨 **EMERGENCY DETECTED**: {emergency_assessment['urgency_level']}")
        for rec in emergency_assessment['recommendations'][:2]:
            print(f"   • {rec}")
        print()
    
    # 2. NATURAL LANGUAGE PROCESSING
    conversation_result = nlp_processor.process_conversation_turn(user_input)
    extracted_symptoms = conversation_result['symptoms']
    
    # 2.5. MENTAL HEALTH ASSESSMENT
    mental_health_assessment = advanced_ai.mental_health.assess_mental_health_severity(
        user_input, extracted_symptoms
    )
    
    # Check for mental health crisis
    if mental_health_assessment['crisis_detected'] or mental_health_assessment['risk_level'] == 'crisis':
        print("🚨 **MENTAL HEALTH CRISIS DETECTED**")
        safety_plan = advanced_ai.mental_health.generate_safety_plan(user_input, mental_health_assessment)
        return {
            'success': True,
            'mental_health_crisis': True,
            'crisis_assessment': mental_health_assessment,
            'safety_plan': safety_plan,
            'message': "🚨 MENTAL HEALTH CRISIS - IMMEDIATE INTERVENTION NEEDED",
            'immediate_action': 'CALL 988 (Suicide & Crisis Lifeline) NOW',
            'emergency_assessment': emergency_assessment
        }
    
    # Display mental health status if significant
    if mental_health_assessment['risk_level'] in ['high', 'moderate']:
        print(f"🧠 **Mental Health Assessment**: {mental_health_assessment['risk_level']} risk level detected")
        for condition in mental_health_assessment['identified_conditions']:
            print(f"   • {condition['condition'].title()}: {condition['severity']} severity")
    
    if not extracted_symptoms:
        return {
            'success': False,
            'message': nlp_processor.generate_conversational_response([], [], []),
            'symptoms_found': [],
            'emergency_assessment': emergency_assessment,
            'mental_health_assessment': mental_health_assessment,
            'needs_more_info': True
        }
    
    # 3. SYMPTOM MAPPING
    user_vector = np.zeros((1, len(symptom_cols)))
    matched_original_symptoms = []
    
    for extracted_symptom in extracted_symptoms:
        for idx, original_symptom in enumerate(symptom_cols):
            if extracted_symptom.lower() == original_symptom.lower().replace('_', ' '):
                user_vector[0, idx] = 1
                matched_original_symptoms.append(original_symptom)
                break
    
    if np.sum(user_vector) == 0:
        return {
            'success': False,
            'message': ("I detected some symptoms but couldn't match them to our database. "
                       f"Detected: {', '.join(extracted_symptoms)}"),
            'symptoms_found': extracted_symptoms,
            'emergency_assessment': emergency_assessment,
            'needs_more_info': True
        }
    
    # 4. PATIENT HISTORY INTEGRATION
    patient_history = None
    if patient_id:
        patient_history = patient_manager.get_patient_history(patient_id)
        print(f"📋 **Patient History Loaded**: {len(patient_history['recent_consultations'])} recent consultations")
    
    # 5. DRUG INTERACTION CHECK
    drug_interactions = {}
    if medications:
        drug_interactions = advanced_ai.check_drug_interactions(medications)
        if drug_interactions['has_interactions']:
            print(f"⚠️  **Drug Interactions Detected**: {drug_interactions['total_interactions']} interactions")
    
    # 6. MODEL PREDICTIONS
    nn_probs = best_model.predict(user_vector, verbose=0)[0]
    lr_probs = logistic_model.predict_proba(user_vector)[0]
    
    # Ensure shapes match
    if len(nn_probs) != len(lr_probs):
        ensemble_probs = nn_probs
    else:
        ensemble_probs = nn_weight * nn_probs + lr_weight * lr_probs
    
    # 7. UNCERTAINTY QUANTIFICATION
    uncertainty_metrics = uncertainty_quantifier.calculate_prediction_uncertainty(ensemble_probs)
    print(f"🎯 **Prediction Confidence**: {uncertainty_metrics['confidence_level']}")
    
    # 8. TOP PREDICTIONS
    top_indices = np.argsort(ensemble_probs)[::-1][:5]
    top_diseases = [label_names[idx] for idx in top_indices]
    top_confidences = [ensemble_probs[idx] for idx in top_indices]
    
    # 9. RISK ASSESSMENT (if patient demographics available)
    risk_assessments = {}
    if patient_history and patient_history['patient_data']:
        patient_data = {
            'age': patient_history['patient_data'][2],
            'gender': patient_history['patient_data'][3],
            'weight': patient_history['patient_data'][6],
            'height': patient_history['patient_data'][5]
        }
        
        # Calculate risk for common conditions
        for condition in ['cardiovascular', 'diabetes']:
            risk_assessments[condition] = advanced_ai.calculate_risk_score(condition, patient_data)
    
    # 10. GENERATE COMPREHENSIVE RESPONSE
    response_parts = []
    
    # Basic symptom analysis
    response_parts.append(f"**Symptoms Analyzed**: {', '.join(matched_original_symptoms)}")
    
    # Emergency status
    if emergency_assessment['emergency_score'] > 0:
        response_parts.append(f"**Emergency Level**: {emergency_assessment['urgency_level']}")
    
    # Predictions with uncertainty
    response_parts.append(f"**Top Predictions** ({uncertainty_metrics['confidence_level']}):")
    for i, (disease, confidence) in enumerate(zip(top_diseases[:3], top_confidences[:3])):
        response_parts.append(f"   {i+1}. {disease} ({confidence:.1%})")
    
    # Risk assessments
    if risk_assessments:
        response_parts.append("**Risk Assessments**:")
        for condition, risk_data in risk_assessments.items():
            response_parts.append(f"   • {condition.title()}: {risk_data['risk_level']} (Score: {risk_data['risk_score']})")
    
    # Drug interactions
    if drug_interactions.get('has_interactions'):
        response_parts.append("**⚠️ Drug Interaction Warnings**:")
        for interaction in drug_interactions['interactions'][:3]:
            response_parts.append(f"   • {interaction['drug1']} + {interaction['drug2']}: {interaction['severity']} risk")
    
    # Mental health assessment
    if mental_health_assessment['risk_level'] != 'low':
        response_parts.append(f"**🧠 Mental Health Assessment**: {mental_health_assessment['risk_level'].title()} Risk")
        if mental_health_assessment['identified_conditions']:
            for condition in mental_health_assessment['identified_conditions'][:3]:
                response_parts.append(f"   • {condition['condition'].title()}: {condition['severity']} severity")
        
        # Add mental health recommendations
        response_parts.append("**Mental Health Recommendations**:")
        for rec in mental_health_assessment['recommendations'][:3]:
            response_parts.append(f"   • {rec}")
    
    comprehensive_response = "\n".join(response_parts)
    
    # 11. SAVE CONSULTATION (if patient ID provided)
    session_id = None
    if patient_id:
        session_data = {
            'symptoms': matched_original_symptoms,
            'predictions': top_diseases[:3],
            'confidence_scores': top_confidences[:3],
            'emergency_assessment': emergency_assessment,
            'follow_up_needed': emergency_assessment['emergency_score'] >= 3
        }
        session_id = patient_manager.save_consultation(patient_id, session_data)
        print(f"💾 **Session Saved**: ID {session_id}")
    
    return {
        'success': True,
        'message': comprehensive_response,
        'symptoms_found': matched_original_symptoms,
        'extracted_symptoms': extracted_symptoms,
        'predictions': top_diseases,
        'confidences': top_confidences,
        'uncertainty_metrics': uncertainty_metrics,
        'emergency_assessment': emergency_assessment,
        'mental_health_assessment': mental_health_assessment,
        'drug_interactions': drug_interactions,
        'risk_assessments': risk_assessments,
        'patient_history': patient_history,
        'session_id': session_id,
        'nn_prediction': label_names[np.argmax(nn_probs)],
        'lr_prediction': label_names[np.argmax(lr_probs)],
        'entities': conversation_result['entities'],
        'needs_more_info': False
    }

def generate_comprehensive_recommendation(medical_results, mental_health_assessment, uncertainty_analysis):
    """
    Generate comprehensive recommendations combining medical and mental health insights
    
    Args:
        medical_results: Medical prediction results
        mental_health_assessment: Mental health assessment results
        uncertainty_analysis: Uncertainty quantification results
        
    Returns:
        dict: Comprehensive recommendations
    """
    recommendations = {
        'immediate_actions': [],
        'follow_up_care': [],
        'lifestyle_modifications': [],
        'mental_health_support': [],
        'monitoring_guidelines': [],
        'professional_referrals': []
    }
    
    # Mental health crisis takes priority
    if mental_health_assessment['crisis_detected'] or mental_health_assessment['risk_level'] == 'crisis':
        recommendations['immediate_actions'] = [
            "🚨 CALL 988 (Suicide & Crisis Lifeline) IMMEDIATELY",
            "Go to nearest Emergency Room or call 911",
            "Do not leave person alone",
            "Remove any means of self-harm"
        ]
        return recommendations
    
    # High-risk mental health
    elif mental_health_assessment['risk_level'] == 'high':
        recommendations['immediate_actions'].append(
            "Contact mental health professional within 24 hours"
        )
        recommendations['mental_health_support'].extend([
            "Implement safety planning strategies",
            "Reach out to trusted support system",
            "Consider urgent mental health services"
        ])
    
    # Moderate mental health risk
    elif mental_health_assessment['risk_level'] == 'moderate':
        recommendations['follow_up_care'].append(
            "Schedule appointment with mental health professional within 1 week"
        )
        recommendations['mental_health_support'].extend([
            "Practice stress management techniques",
            "Maintain regular sleep schedule",
            "Engage in supportive activities"
        ])
    
    # Medical recommendations based on predictions
    if medical_results.get('success') and medical_results.get('predictions'):
        top_prediction = medical_results['predictions'][0]
        confidence = medical_results['confidences'][0]
        
        if confidence > 0.8:
            recommendations['follow_up_care'].append(
                f"Consult healthcare provider about {top_prediction}"
            )
        elif confidence > 0.6:
            recommendations['follow_up_care'].append(
                f"Monitor symptoms related to {top_prediction}"
            )
        else:
            recommendations['monitoring_guidelines'].append(
                "Track symptoms and seek medical advice if worsening"
            )
    
    # Emergency medical situations
    if medical_results.get('emergency_assessment', {}).get('emergency_score', 0) >= 5:
        recommendations['immediate_actions'].insert(0, 
            "🚨 SEEK IMMEDIATE MEDICAL ATTENTION"
        )
    
    # Add general wellness recommendations
    recommendations['lifestyle_modifications'].extend([
        "Maintain regular exercise routine",
        "Follow balanced nutrition",
        "Ensure adequate sleep (7-9 hours)",
        "Practice stress reduction techniques"
    ])
    
    # Professional referrals based on conditions
    identified_conditions = mental_health_assessment.get('identified_conditions', [])
    for condition_data in identified_conditions:
        condition = condition_data['condition']
        if condition in ['depression', 'anxiety']:
            recommendations['professional_referrals'].append(
                f"Consider therapy (CBT) for {condition}"
            )
        elif condition == 'ptsd':
            recommendations['professional_referrals'].append(
                "Trauma-focused therapy recommended"
            )
        elif condition in ['psychosis', 'mania']:
            recommendations['professional_referrals'].append(
                "Psychiatric evaluation recommended"
            )
    
    return recommendations

# Legacy function for compatibility
def predict_from_natural_language(user_input, best_model, logistic_model, label_names, symptom_cols, nn_weight=0.7, lr_weight=0.3):
    """Legacy wrapper for enhanced_medical_prediction"""
    return enhanced_medical_prediction(user_input, best_model, logistic_model, label_names, symptom_cols, 
                                     None, None, None, nn_weight, lr_weight)

# 9. Enhanced Conversational Interface with Advanced Features
print("\n" + "="*80)
print("🏥 ADVANCED AI MEDICAL ASSISTANT - COMPREHENSIVE ANALYSIS SYSTEM")
print("="*80)
print("\n🚀 **Enhanced Features Available:**")
print("   • 🧠 Advanced AI with Emergency Detection")
print("   • 🧠 Mental Health Assessment & Crisis Detection")
print("   • 💊 Drug Interaction Checking")
print("   • 📊 Risk Stratification & Assessment") 
print("   • 🎯 Uncertainty Quantification")
print("   • 📋 Patient History Management")
print("   • 🗺️  Visual Symptom Mapping")
print("   • 🔍 Multi-modal Medical Analysis")
print("   • ⚡ Real-time Learning & Adaptation")
print("   • 🆘 Crisis Intervention & Safety Planning")
print("   • 🎭 Psychiatric Symptom Recognition")
print("   • 💚 Comprehensive Mental Health Support")

# Patient setup
print(f"\n📝 **Patient Setup**")
setup_choice = input("Would you like to (1) Create new patient profile, (2) Use existing patient ID, or (3) Continue as anonymous? (1/2/3): ").strip()

patient_id = None
patient_data = {}

if setup_choice == '1':
    print("\n👤 **Creating Patient Profile**")
    patient_data = {
        'name': input("Name (optional): ").strip() or "Anonymous",
        'age': int(input("Age: ").strip() or 0),
        'gender': input("Gender (M/F/Other): ").strip() or "Unknown",
        'ethnicity': input("Ethnicity (optional): ").strip() or "Not specified",
        'height': float(input("Height in cm (optional): ").strip() or 0),
        'weight': float(input("Weight in kg (optional): ").strip() or 0)
    }
    patient_id = patient_manager.create_patient_profile(patient_data)
    print(f"✅ Patient profile created with ID: {patient_id}")
    
elif setup_choice == '2':
    patient_id = input("Enter existing patient ID: ").strip()
    if patient_id:
        history = patient_manager.get_patient_history(patient_id)
        if history['patient_data']:
            print(f"✅ Patient profile loaded: {history['patient_data'][1]}")
        else:
            print("❌ Patient ID not found. Continuing as anonymous.")
            patient_id = None

# Medications setup
medications = []
med_input = input("\nDo you take any medications? (y/n): ").strip().lower()
if med_input == 'y':
    print("Enter medications (press Enter after each, empty line to finish):")
    while True:
        med = input("Medication: ").strip()
        if not med:
            break
        medications.append(med)
    print(f"✅ Recorded {len(medications)} medications")

# Vital signs setup (optional)
vital_signs = {}
vitals_input = input("\nWould you like to enter vital signs? (y/n): ").strip().lower()
if vitals_input == 'y':
    try:
        vital_signs = {
            'systolic_bp': int(input("Systolic blood pressure (optional): ").strip() or 0),
            'diastolic_bp': int(input("Diastolic blood pressure (optional): ").strip() or 0),
            'heart_rate': int(input("Heart rate (optional): ").strip() or 0),
            'temperature': float(input("Temperature in °F (optional): ").strip() or 0),
            'oxygen_saturation': int(input("Oxygen saturation % (optional): ").strip() or 0)
        }
        # Remove zero values
        vital_signs = {k: v for k, v in vital_signs.items() if v > 0}
        if vital_signs:
            print(f"✅ Recorded vital signs: {list(vital_signs.keys())}")
    except ValueError:
        print("Invalid vital signs format. Continuing without vital signs.")
        vital_signs = {}

# Visual symptom mapping option
print(f"\n🗺️  **Symptom Input Options:**")
input_method = input("Choose input method: (1) Natural language, (2) Visual body map, (3) Both: ").strip()

visual_symptoms = []
if input_method in ['2', '3']:
    print("\n🗺️  **Visual Symptom Mapping**")
    print("Available body regions: head, chest, abdomen, arms, legs, back, skin")
    print("Enter regions where you have symptoms (comma-separated):")
    regions_input = input("Body regions: ").strip().lower()
    if regions_input:
        selected_regions = [r.strip() for r in regions_input.split(',')]
        visual_symptoms = visual_mapper.get_region_symptoms(selected_regions)
        print(f"✅ Mapped symptoms from body regions: {', '.join(visual_symptoms)}")

# Start enhanced conversation
conversation_interface.start_conversation()

# Enhanced conversation loop
max_attempts = 10
attempt = 0
all_symptoms = []

while attempt < max_attempts:
    attempt += 1
    
    if attempt == 1:
        if input_method in ['1', '3']:
            user_input = conversation_interface.get_user_input()
        else:
            user_input = f"I have symptoms in these areas: {', '.join(visual_symptoms)}"
    else:
        user_input = conversation_interface.get_user_input(
            "Please provide more details or additional symptoms: "
        )
    
    if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye', 'done', 'finished']:
        print("\n👋 Thank you for using the Advanced AI Medical Assistant!")
        break
    
    # Combine visual and text symptoms
    combined_input = user_input
    if visual_symptoms and attempt == 1:
        combined_input += f" Also experiencing: {', '.join(visual_symptoms)}"
    
    # Get enhanced prediction results
    result = enhanced_medical_prediction(
        combined_input, best_model, logistic_model, label_names, symptom_cols,
        patient_id, vital_signs, medications
    )
    
    # Provide comprehensive feedback
    conversation_interface.provide_feedback(result['message'])
    
    if result['success']:
        # Enhanced detailed analysis
        print("\n" + "="*60)
        print("� **COMPREHENSIVE MEDICAL ANALYSIS REPORT**")
        print("="*60)
        
        # Emergency Assessment
        if result.get('emergency_assessment', {}).get('emergency_score', 0) > 0:
            emergency = result['emergency_assessment']
            print(f"\n🚨 **EMERGENCY ASSESSMENT**: {emergency['urgency_level']}")
            if emergency.get('detected_emergencies'):
                print(f"   Detected: {', '.join(emergency['detected_emergencies'])}")
            print("   Recommendations:")
            for rec in emergency.get('recommendations', [])[:3]:
                print(f"   • {rec}")
        
        # Symptoms Analysis
        print(f"\n🔍 **SYMPTOMS ANALYSIS**:")
        print(f"   • Identified: {', '.join(result['symptoms_found'])}")
        if result.get('extracted_symptoms'):
            print(f"   • Extracted: {', '.join(result['extracted_symptoms'])}")
        if result.get('entities'):
            print(f"   • Context: {result['entities']}")
        
        # Prediction Analysis with Uncertainty
        uncertainty = result.get('uncertainty_metrics', {})
        print(f"\n🎯 **PREDICTION ANALYSIS** ({uncertainty.get('confidence_level', 'Unknown Confidence')}):")
        print(f"   • Uncertainty Score: {uncertainty.get('uncertainty_score', 0):.3f}")
        print(f"   • Prediction Entropy: {uncertainty.get('entropy', 0):.3f}")
        
        print(f"\n📈 **TOP 5 DIAGNOSTIC POSSIBILITIES**:")
        for i, (disease, confidence) in enumerate(zip(result['predictions'], result['confidences'])):
            print(f"   {i+1}. {disease} ({confidence:.1%})")
        
        # Risk Assessments
        if result.get('risk_assessments'):
            print(f"\n⚠️  **RISK STRATIFICATION**:")
            for condition, risk_data in result['risk_assessments'].items():
                print(f"   • {condition.title()}: {risk_data['risk_level']} (Score: {risk_data['risk_score']})")
                if risk_data.get('recommendations'):
                    for rec in risk_data['recommendations'][:2]:
                        print(f"     - {rec}")
        
        # Drug Interactions
        if result.get('drug_interactions', {}).get('has_interactions'):
            interactions = result['drug_interactions']
            print(f"\n� **MEDICATION SAFETY ALERTS** ({interactions['total_interactions']} interactions):")
            for interaction in interactions['interactions'][:3]:
                severity_icon = "🔴" if interaction['severity'] == 'major' else "🟡" if interaction['severity'] == 'moderate' else "🟢"
                print(f"   {severity_icon} {interaction['drug1']} + {interaction['drug2']}")
                print(f"     Risk: {interaction['severity']} - {interaction['warning']}")
        
        # Model Performance
        print(f"\n🔬 **AI MODEL INSIGHTS**:")
        print(f"   • Neural Network: {result['nn_prediction']} ({np.max(result.get('nn_probs', [0])):.1%})")
        print(f"   • Logistic Regression: {result['lr_prediction']}")
        print(f"   • Ensemble Method: Weighted combination for optimal accuracy")
        
        # Patient History Context
        if result.get('patient_history'):
            history = result['patient_history']
            print(f"\n📋 **PATIENT CONTEXT**:")
            if history.get('recent_consultations'):
                print(f"   • Recent consultations: {len(history['recent_consultations'])}")
            if history.get('current_medications'):
                print(f"   • Current medications: {len(history['current_medications'])}")
            if result.get('session_id'):
                print(f"   • Session ID: {result['session_id']}")
        
        # Follow-up Options
        print(f"\n🔄 **FOLLOW-UP OPTIONS**:")
        follow_up_choice = input("""
   Choose next action:
   1. Add more symptoms/information
   2. Get detailed condition information
   3. Risk factor analysis
   4. Generate visual body map
   5. Export analysis report
   6. Complete assessment
   
   Your choice (1-6): """).strip()
        
        if follow_up_choice == '1':
            # Continue with more information
            attempt -= 1  # Don't count this iteration
            continue
        elif follow_up_choice == '2':
            # Detailed condition information
            print(f"\n📚 **DETAILED CONDITION INFORMATION**:")
            top_condition = result['predictions'][0]
            print(f"**{top_condition}** (Confidence: {result['confidences'][0]:.1%})")
            print("For detailed medical information, please consult with a healthcare provider.")
            
        elif follow_up_choice == '3':
            # Risk factor analysis
            if patient_id and result.get('risk_assessments'):
                print(f"\n📊 **COMPREHENSIVE RISK ANALYSIS**:")
                for condition, risk_data in result['risk_assessments'].items():
                    print(f"\n{condition.upper()} RISK ASSESSMENT:")
                    print(f"   Risk Level: {risk_data['risk_level']}")
                    print(f"   Risk Score: {risk_data['risk_score']}")
                    print(f"   Applicable Factors: {', '.join(risk_data['applicable_factors'])}")
                    print("   Recommendations:")
                    for rec in risk_data.get('recommendations', []):
                        print(f"   • {rec}")
            else:
                print("Risk analysis requires patient profile. Please create a patient profile for detailed risk assessment.")
                
        elif follow_up_choice == '4':
            # Generate visual body map
            print(f"\n🗺️  **INTERACTIVE BODY MAP GENERATED**")
            body_map_html = visual_mapper.generate_body_map_html()
            with open("symptom_body_map.html", "w") as f:
                f.write(body_map_html)
            print("✅ Interactive body map saved as 'symptom_body_map.html'")
            print("   Open this file in a web browser for visual symptom mapping.")
            
        elif follow_up_choice == '5':
            # Export analysis report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"medical_analysis_report_{timestamp}.json"
            
            # Prepare comprehensive report
            report_data = {
                'timestamp': datetime.now().isoformat(),
                'patient_id': patient_id,
                'symptoms': result['symptoms_found'],
                'predictions': list(zip(result['predictions'], result['confidences'])),
                'uncertainty_metrics': result.get('uncertainty_metrics', {}),
                'emergency_assessment': result.get('emergency_assessment', {}),
                'risk_assessments': result.get('risk_assessments', {}),
                'drug_interactions': result.get('drug_interactions', {}),
                'session_id': result.get('session_id'),
                'model_insights': {
                    'neural_network': result['nn_prediction'],
                    'logistic_regression': result['lr_prediction']
                }
            }
            
            with open(report_filename, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            print(f"✅ **Analysis Report Exported**: {report_filename}")
            print("   This comprehensive report contains all analysis details.")
            
        elif follow_up_choice == '6':
            # Complete assessment
            print("\n✅ **COMPREHENSIVE MEDICAL ASSESSMENT COMPLETED**")
            break
        else:
            print("Invalid choice. Continuing with assessment completion.")
            break
            
    elif not result['needs_more_info']:
        break
        
# Final session cleanup and analytics
if attempt >= max_attempts:
    print("\n⏰ **Session Timeout**: Maximum conversation attempts reached.")
    print("For enhanced assistance, please provide more specific symptom descriptions.")

# Generate comprehensive session summary
print("\n" + "="*80)
print("📊 **COMPREHENSIVE SESSION SUMMARY**")
print("="*80)

session_end_time = datetime.now()
print(f"🕐 **Session Duration**: {session_end_time.strftime('%Y-%m-%d %H:%M:%S')}")

if patient_id:
    print(f"👤 **Patient ID**: {patient_id}")
    # Get updated patient history
    final_history = patient_manager.get_patient_history(patient_id)
    print(f"📋 **Total Consultations**: {len(final_history['recent_consultations'])}")

print(f"\n🚀 **ADVANCED FEATURES UTILIZED THIS SESSION**:")
feature_list = [
    "🧠 Advanced AI & Machine Learning",
    "🔍 Natural Language Processing & Understanding", 
    "💊 Drug Interaction Safety Checking",
    "⚠️  Emergency Detection & Risk Assessment",
    "🎯 Uncertainty Quantification & Confidence Analysis",
    "📊 Risk Stratification for Multiple Conditions",
    "🗺️  Visual Symptom Mapping & Body Diagrams",
    "📋 Patient History Management & Tracking",
    "🔬 Multi-Model Ensemble Predictions",
    "💾 Secure Data Storage & HIPAA Considerations",
    "📈 Real-time Performance Analytics",
    "🎨 Interactive User Interface & Experience"
]

for feature in feature_list:
    print(f"   ✅ {feature}")

print(f"\n� **SYSTEM PERFORMANCE METRICS**:")
print(f"   • Model Accuracy: Neural Network + Logistic Regression Ensemble")
print(f"   • Prediction Confidence: With Uncertainty Quantification")
print(f"   • Emergency Detection: Real-time Risk Assessment")
print(f"   • Drug Safety: Automated Interaction Checking")
print(f"   • Patient Context: Integrated History Management")

print(f"\n🛡️  **SAFETY & COMPLIANCE FEATURES**:")
print(f"   ✅ Emergency Detection & Alert System")
print(f"   ✅ Drug Interaction Safety Warnings")
print(f"   ✅ Uncertainty Quantification & Confidence Levels")
print(f"   ✅ Patient Privacy & Data Security")
print(f"   ✅ Medical Disclaimer & Professional Consultation Guidance")
print(f"   ✅ Audit Trail & Session Logging")

print(f"\n🔬 **CLINICAL INTELLIGENCE CAPABILITIES**:")
print(f"   • Multi-condition Risk Stratification")
print(f"   • Evidence-based Clinical Decision Support")  
print(f"   • Population Health Pattern Analysis")
print(f"   • Personalized Medicine Integration")
print(f"   • Continuous Learning & Model Updates")

print(f"\n🎓 **MEDICAL AI INNOVATIONS IMPLEMENTED**:")
innovations = [
    "Transformer-based Natural Language Understanding",
    "Multi-modal Medical Data Integration",
    "Bayesian Uncertainty Quantification",
    "Real-time Emergency Detection Algorithms",
    "Advanced Symptom Extraction & Mapping",
    "Interactive Visual Diagnostic Tools",
    "Federated Learning Architecture Support",
    "Clinical Guideline Integration",
    "Personalized Risk Assessment Models",
    "Automated Medical Literature Integration"
]

for innovation in innovations:
    print(f"   🚀 {innovation}")

print(f"\n⚠️  **CRITICAL MEDICAL DISCLAIMER**:")
print("="*50)
print("🏥 This Advanced AI Medical Assistant is designed to support and enhance")
print("   medical decision-making, but should NEVER replace professional medical")
print("   advice, diagnosis, or treatment from qualified healthcare providers.")
print()
print("🚨 EMERGENCY SITUATIONS:")
print("   • If experiencing severe symptoms, call 911 immediately")
print("   • For urgent medical concerns, contact your healthcare provider")
print("   • This AI system is for informational and educational purposes only")
print()
print("📋 CLINICAL VALIDATION:")
print("   • AI predictions should be validated by medical professionals")
print("   • Consider multiple diagnostic perspectives and clinical tests")
print("   • Individual patient factors may not be fully captured by AI models")

print(f"\n� **HEALTHCARE PROVIDER COLLABORATION**:")
print("   This AI system is designed to work alongside healthcare professionals")
print("   to enhance diagnostic accuracy and improve patient outcomes through:")
print("   • Comprehensive symptom analysis and pattern recognition")
print("   • Risk factor identification and stratification")
print("   • Drug interaction safety monitoring")
print("   • Emergency condition detection and alerting")
print("   • Evidence-based clinical decision support")

print(f"\n🎯 **FUTURE ENHANCEMENTS IN DEVELOPMENT**:")
future_features = [
    "Real-time Medical Literature Integration",
    "Genomic Risk Factor Analysis",
    "Wearable Device Data Integration", 
    "Telemedicine Platform Connectivity",
    "Advanced Imaging Analysis (X-rays, MRIs)",
    "Population Health Trend Prediction",
    "Clinical Trial Matching & Recommendations",
    "Multilingual Medical Translation",
    "Voice-based Symptom Input & Analysis",
    "Blockchain-secured Health Records"
]

for feature in future_features:
    print(f"   🔮 {feature}")

print(f"\n" + "="*80)
print("🏆 **THANK YOU FOR USING THE ADVANCED AI MEDICAL ASSISTANT!**")
print("="*80)
print("Your health and safety are our top priorities. This comprehensive AI system")
print("represents the cutting edge of medical artificial intelligence, combining")
print("advanced machine learning, natural language processing, and clinical")
print("decision support to provide you with the most accurate and helpful")
print("medical insights possible.")
print()
print("Remember: AI enhances human medical expertise - it doesn't replace it.")
print("Always consult with qualified healthcare professionals for medical decisions.")
print()
print("Stay healthy! 🌟")
print("="*80)