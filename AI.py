import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info/warning messages
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split


# 1. Load and clean the real dataset
df = pd.read_csv('C:/Users/bubhi/OneDrive/Desktop/AI/tfenv311/Training.csv')
# Remove columns with all zeros and unnamed columns
drop_cols = [col for col in df.columns if col.startswith('Unnamed') or df[col].sum() == 0]
df = df.drop(columns=drop_cols)
# Fill missing values with 0 (assume symptom not present)
df = df.fillna(0)

# 2. Prepare features and labels
symptom_cols = [col for col in df.columns if col != 'prognosis']
X = df[symptom_cols].values
y = pd.factorize(df['prognosis'])[0]
label_names = pd.factorize(df['prognosis'])[1]

# 3. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 4. Define a function to build the model (for cross-validation and tuning)
def build_model(optimizer='adam'):
    model = models.Sequential([
        layers.Input(shape=(len(symptom_cols),)),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(len(label_names), activation='softmax')
    ])
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# 5. Model training/loading logic
model_path = 'disease_model.h5'
if os.path.exists(model_path):
    best_model = load_model(model_path)
    print("Loaded trained model from file.")
    # For reporting, set dummy best_params
    best_params = (16, 'adam')
else:
    from sklearn.model_selection import StratifiedKFold
    from tensorflow.keras.callbacks import EarlyStopping
    import numpy as np
    best_val_acc = 0
    best_params = None
    best_model = None
    batch_sizes = [16, 32]
    optimizers = ['adam', 'rmsprop']
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for batch_size in batch_sizes:
        for optimizer in optimizers:
            val_accuracies = []
            for train_idx, val_idx in kfold.split(X_train, y_train):
                model = build_model(optimizer=optimizer)
                early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                model.fit(
                    X_train[train_idx], y_train[train_idx],
                    epochs=100,
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
                best_model = build_model(optimizer=optimizer)
                early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                best_model.fit(
                    X_train, y_train,
                    epochs=100,
                    batch_size=batch_size,
                    validation_split=0.1,
                    verbose=0,
                    callbacks=[early_stop]
                )
    # Save the trained model
    best_model.save(model_path)
    print(f"Model trained and saved to {model_path}.")

# 6. Evaluate the best model
loss, accuracy = best_model.evaluate(X_test, y_test, verbose=0)
print(f"\nBest params: batch_size={best_params[0]}, optimizer={best_params[1]}")
print(f"Test accuracy: {accuracy:.2f}")

# 7. Prompt user for symptoms

# User-friendly symptom input prompt
print("\nWelcome to the AI Disease Predictor!")
print("You can enter your symptoms and the AI will predict the most likely disease.")
print("\nHere are some example symptoms you can use:")
example_symptoms = ', '.join(symptom_cols[:10]) + ', ...'
print(example_symptoms)
print("\nType your symptoms separated by commas (e.g. fever, headache, nausea)")

while True:
    user_input = input("Enter your symptoms: ").strip().lower()
    user_symptoms = [s.strip() for s in user_input.split(',') if s.strip()]
    # Validate at least one valid symptom
    valid = any(symptom.lower() in [col.lower() for col in symptom_cols] for symptom in user_symptoms)
    if not user_symptoms or not valid:
        print("\nPlease enter at least one valid symptom from the list. Try again.")
        print("Example symptoms:", example_symptoms)
    else:
        break

# 8. Create input vector for user
user_vector = np.zeros((1, len(symptom_cols)))
for idx, symptom in enumerate(symptom_cols):
    if symptom.lower() in user_symptoms:
        user_vector[0, idx] = 1

# 9. Predict disease and show top 3 most likely
pred = best_model.predict(user_vector)
top_indices = np.argsort(pred[0])[::-1][:3]
print("\nTop 3 predicted diseases:")
for rank, idx in enumerate(top_indices, 1):
    print(f"{rank}. {label_names[idx]} (probability: {pred[0][idx]:.2f})")

# Print all probabilities for transparency
print("\nAll disease probabilities:")
for idx, name in enumerate(label_names):
    print(f"{name}: {pred[0][idx]:.4f}")