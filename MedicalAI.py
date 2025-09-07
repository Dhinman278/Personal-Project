import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
import joblib

class BaseAI:
    def __init__(self, name="MyAI"):
        self.name = name

    def train(self, data):
        pass

    def predict(self, input_data):
        return "This is a base response."

    def save(self, filepath):
        pass

    def load(self, filepath):
        pass

class MedicalDiagnosisAI(BaseAI):
    def __init__(self, name="MedicalAI"):
        super().__init__(name)
        self.model = None

    def train(self, csv_path):
        # Load dataset
        df = pd.read_csv(csv_path)
        X = df['diagnosis']  # e.g., "fever,cough"
        y = df['symptoms'] # e.g., "flu"
        # Create a pipeline: vectorize symptoms, then classify
        self.model = Pipeline([
            ('vect', CountVectorizer()),
            ('clf', DecisionTreeClassifier())
        ])
        self.model.fit(X, y)

    def predict(self, symptoms):
        if not self.model:
            return "Model not trained."
        return self.model.predict([symptoms])[0]

    def save(self, filepath):
        if self.model:
            joblib.dump(self.model, filepath)

    def load(self, filepath):
        self.model = joblib.load(filepath)

# Example usage:
if __name__ == "__main__":
    ai = MedicalDiagnosisAI("NurseHelper")
    ai.train("patient_data")  # Make sure the file is in the same folder as this script
    # response = ai.predict("fever, cough")
    # print(response)