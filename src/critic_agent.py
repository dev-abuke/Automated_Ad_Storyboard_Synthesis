import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

class CriticGradingAgent:
    def __init__(self):
        # These weights are based on the correlations we found
        self.er_weights = {20: 0.096262, 11: 0.069027, 50: 0.064961}
        self.ctr_weights = {67: 0.087513, 20: 0.084923, 87: 0.079520}
        self.scaler = joblib.load('feature_scaler.joblib')  # Load the scaler we used for original features

    def extract_features(self, image):
        # Placeholder for feature extraction
        # In a real scenario, this would use the same feature extraction process as our training data
        return np.random.rand(100)  # Assuming 100 features for this example

    def score_er(self, features):
        return sum(features[k] * v for k, v in self.er_weights.items())

    def score_ctr(self, features):
        return sum(features[k] * v for k, v in self.ctr_weights.items())

    def overall_score(self, er_score, ctr_score):
        return 0.5 * er_score + 0.5 * ctr_score  # Equal weight to ER and CTR

    def generate_feedback(self, er_score, ctr_score, overall_score):
        feedback = []
        if overall_score > 0.7:
            feedback.append("This ad shows strong potential for both engagement and clicks.")
        elif overall_score > 0.5:
            feedback.append("This ad has moderate potential but could be improved.")
        else:
            feedback.append("This ad may struggle to engage users and generate clicks.")

        if er_score > ctr_score:
            feedback.append("The ad seems better at engaging users than generating clicks. Consider strengthening the call-to-action.")
        elif ctr_score > er_score:
            feedback.append("The ad seems effective at generating clicks but might not engage users as much. Consider making the content more engaging.")

        return " ".join(feedback)

    def evaluate_ad(self, image):
        features = self.extract_features(image)
        scaled_features = self.scaler.transform([features])[0]
        
        er_score = self.score_er(scaled_features)
        ctr_score = self.score_ctr(scaled_features)
        overall_score = self.overall_score(er_score, ctr_score)
        
        feedback = self.generate_feedback(er_score, ctr_score, overall_score)
        
        return {
            "ER Score": er_score,
            "CTR Score": ctr_score,
            "Overall Score": overall_score,
            "Feedback": feedback
        }

# Example usage
if __name__ == "__main__":
    agent = CriticGradingAgent()
    
    # Simulate evaluating multiple ad images
    for i in range(5):
        print(f"\nEvaluating Ad {i+1}")
        result = agent.evaluate_ad(None)  # Passing None as we're using random features for this example
        for key, value in result.items():
            print(f"{key}: {value}")