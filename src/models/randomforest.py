from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import log_loss

class RandomForestModel:
    def __init__(self, n_estimators=100):
        self.classifier = RandomForestClassifier(n_estimators=n_estimators)

    def train(self, features, labels):
        self.classifier.fit(features, labels)
        predictions = self.classifier.predict(features)
        accuracy = self.compute_accuracy(predictions, labels)
        loss = log_loss(labels, self.classifier.predict_proba(features))
        return loss, accuracy
         

    def predict(self, features):
        predictions = self.classifier.predict(features)
        return predictions

    def test(self, features, labels):
        predictions = self.predict(features)
        accuracy = self.compute_accuracy(predictions, labels)
        loss = log_loss(labels, self.classifier.predict_proba(features))
        return predictions, accuracy, loss

    @staticmethod
    def compute_accuracy(predictions, labels):
        correct_predictions = sum(predictions == labels)
        total_predictions = len(labels)
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        return accuracy


