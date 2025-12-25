import os
import joblib

class ExcelIntentClassifier:
    def __init__(self, model_path: str = "excel_intent_model.pkl"):
        self.model_path = model_path
        self.model = None

    def _ensure_model(self):
        if self.model is None:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"Model file '{self.model_path}' not found. Run 'python train_model.py' first."
                )
            self.model = joblib.load(self.model_path)

    def load_model_and_predict(self, text: str) -> str:
        """Legacy method"""
        self._ensure_model()
        return self.model.predict([text])[0]

    def predict(self, text: str) -> str:
        """Cleaner method, matches main.py usage"""
        return self.load_model_and_predict(text)
