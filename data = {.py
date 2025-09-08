import pandas as pd
data = {
    "symptoms": [
        "Fever", "Cough", "Body aches",
        "Increased thirst", "Frequent urination", "Fatigue",
        "Fever", "Cough", "Loss of taste or smell",
        "Headache", "Blurred vision", "Nausea",
        "Shortness of breath", "Wheezing", "Chest tightness",
        "Chronic cough", "Weight loss", "Night sweats",
        "Fever", "Chills", "Sweating",
        "Cough", "Chest pain", "Shortness of breath",
        "Throbbing pain", "Sensitivity to light", "Nausea",
        "Fatigue", "Pale skin", "Shortness of breath",
        "Rash", "Itchy skin", "Fever",
        "Fever", "Rash", "Runny nose",
        "Jaundice", "Fatigue", "Abdominal pain"
    ],
    "diagnosis": [
        "Influenza", "Influenza", "Influenza",
        "Diabetes", "Diabetes", "Diabetes",
        "COVID-19", "COVID-19", "COVID-19",
        "Hypertension", "Hypertension", "Hypertension",
        "Asthma", "Asthma", "Asthma",
        "Tuberculosis", "Tuberculosis", "Tuberculosis",
        "Malaria", "Malaria", "Malaria",
        "Pneumonia", "Pneumonia", "Pneumonia",
        "Migraine", "Migraine", "Migraine",
        "Anemia", "Anemia", "Anemia",
        "Chickenpox", "Chickenpox", "Chickenpox",
        "Measles", "Measles", "Measles",
        "Hepatitis B", "Hepatitis B", "Hepatitis B"
    ],
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("diseases_and_symptoms.csv", index=False)