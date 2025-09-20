#!/usr/bin/env python3
"""
Quick Ensemble AI - Neural Network + Logistic Regression for Disease Prediction
Using the full Training.csv dataset (4920 samples)
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import os

print("🚀 Loading original Training.csv dataset...")
df = pd.read_csv('C:/Users/bubhi/OneDrive/Desktop/AI/tfenv311/Training.csv')
print(f"✅ Dataset shape: {df.shape}")
print(f"✅ Diseases: {df['prognosis'].nunique()} ({df['prognosis'].value_counts().iloc[0]} samples each)")

# Prepare data
symptom_cols = [col for col in df.columns if col != 'prognosis']
X = df[symptom_cols].fillna(0).values
y = pd.factorize(df['prognosis'])[0]
label_names = pd.factorize(df['prognosis'])[1]

print(f"✅ Features: {len(symptom_cols)} symptoms")
print(f"✅ Classes: {len(label_names)} diseases")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 1. Train Logistic Regression (fast)
print("\n🔄 Training Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
lr_model.fit(X_train, y_train)
lr_accuracy = accuracy_score(y_test, lr_model.predict(X_test))
print(f"✅ LR Test Accuracy: {lr_accuracy:.4f}")

# 2. Train Neural Network (simplified)
print("\n🔄 Training Neural Network...")
nn_model = models.Sequential([
    layers.Input(shape=(len(symptom_cols),)),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(label_names), activation='softmax')
])
nn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Quick training with early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
nn_model.fit(X_train, y_train, 
             epochs=30, 
             batch_size=32, 
             validation_split=0.2, 
             verbose=1,
             callbacks=[early_stop])

nn_loss, nn_accuracy = nn_model.evaluate(X_test, y_test, verbose=0)
print(f"✅ NN Test Accuracy: {nn_accuracy:.4f}")

# 3. Create Ensemble
print(f"\n🎯 Creating Ensemble Model...")
nn_probs = nn_model.predict(X_test, verbose=0)
lr_probs = lr_model.predict_proba(X_test)

# Weighted ensemble (70% NN, 30% LR)
ensemble_probs = 0.7 * nn_probs + 0.3 * lr_probs
ensemble_preds = np.argmax(ensemble_probs, axis=1)
ensemble_accuracy = accuracy_score(y_test, ensemble_preds)
print(f"✅ Ensemble Accuracy: {ensemble_accuracy:.4f}")

# Save models for reuse
with open('quick_lr_model.pkl', 'wb') as f:
    pickle.dump(lr_model, f)
nn_model.save('quick_nn_model.h5')
print("✅ Models saved!")

print(f"\n🏆 ENSEMBLE RESULTS:")
print(f"   Logistic Regression: {lr_accuracy:.2%}")
print(f"   Neural Network:      {nn_accuracy:.2%}")
print(f"   Ensemble:            {ensemble_accuracy:.2%}")

# Test with diabetes symptoms
print(f"\n🩺 TESTING WITH DIABETES SYMPTOMS...")
diabetes_symptoms = [
    'blurred_and_distorted_vision',
    'excessive_hunger', 
    'fatigue',
    'increased_appetite',
    'irregular_sugar_level',
    'lethargy',
    'obesity',
    'polyuria',
    'restlessness',
    'weight_loss'
]

# Create symptom vector
user_vector = np.zeros((1, len(symptom_cols)))
matched_symptoms = []
for symptom in diabetes_symptoms:
    if symptom in symptom_cols:
        idx = symptom_cols.index(symptom)
        user_vector[0, idx] = 1
        matched_symptoms.append(symptom)

print(f"📋 Matched symptoms ({len(matched_symptoms)}): {matched_symptoms}")

# Get predictions
nn_pred_probs = nn_model.predict(user_vector, verbose=0)[0]
lr_pred_probs = lr_model.predict_proba(user_vector)[0]
ensemble_pred_probs = 0.7 * nn_pred_probs + 0.3 * lr_pred_probs

# Show top predictions
top_indices = np.argsort(ensemble_pred_probs)[::-1][:5]

print(f"\n🎯 TOP 5 ENSEMBLE PREDICTIONS:")
for i, idx in enumerate(top_indices):
    confidence = ensemble_pred_probs[idx] * 100
    print(f"   {i+1}. {label_names[idx]} - {confidence:.2f}%")

print(f"\n🔍 INDIVIDUAL MODEL PREDICTIONS:")
nn_top = np.argmax(nn_pred_probs)
lr_top = np.argmax(lr_pred_probs)
print(f"   Neural Network:      {label_names[nn_top]} ({nn_pred_probs[nn_top]*100:.2f}%)")
print(f"   Logistic Regression: {label_names[lr_top]} ({lr_pred_probs[lr_top]*100:.2f}%)")

print(f"\n🎉 ENSEMBLE SYSTEM READY! 🎉")
print(f"You can now use this for highly accurate disease predictions!")