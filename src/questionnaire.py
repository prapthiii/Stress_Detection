import numpy as np

class Questionnaire:
    def __init__(self):
        # Perceived Stress Scale (PSS-10) style self-assessment.
        # Items 4, 5, 7, and 8 are reverse-scored.
        self.questions = [
            "In the last month, how often have you been upset because of something that happened unexpectedly?",
            "In the last month, how often have you felt that you were unable to control the important things in your life?",
            "In the last month, how often have you felt nervous and stressed?",
            "In the last month, how often have you felt confident about your ability to handle your personal problems?",
            "In the last month, how often have you felt that things were going your way?",
            "In the last month, how often have you found that you could not cope with all the things you had to do?",
            "In the last month, how often have you been able to control irritations in your life?",
            "In the last month, how often have you felt that you were on top of things?",
            "In the last month, how often have you been angered because of things that were outside of your control?",
            "In the last month, how often have you felt difficulties were piling up so high that you could not overcome them?",
        ]
        self.options = ["Never", "Almost never", "Sometimes", "Fairly often", "Very often"]
        self.scores = [0, 1, 2, 3, 4]
        self.reverse_scored_items = {3, 4, 6, 7}
        self.max_score = 40

    def calculate_score(self, responses):
        """Calculate PSS-10 score from raw option scores."""
        if len(responses) != len(self.questions):
            raise ValueError("Number of responses must match number of questions")
        total_score = 0
        for index, response in enumerate(responses):
            total_score += 4 - response if index in self.reverse_scored_items else response
        return total_score

    def interpret_score(self, score):
        """Interpret PSS-10 score."""
        if score <= 7:
            return "Very low perceived stress", 0
        elif score <= 13:
            return "Low perceived stress", 1
        elif score <= 26:
            return "Moderate perceived stress", 2
        else:
            return "High perceived stress", 3

    def get_features(self, responses):
        """Extract features from questionnaire responses."""
        score = self.calculate_score(responses)
        # Simple features: score, average response, variance
        features = np.array([score, np.mean(responses), np.var(responses)])
        return features
