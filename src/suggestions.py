
import numpy as np

class SupportSuggestions:
    def __init__(self):
        self.suggestions = {
            0: [  # No stress
                "You're doing great! Keep maintaining your positive habits.",
                "Enjoy this stress-free moment.",
                "Stay mindful and continue what works for you."
            ],
            1: [  # Mild stress
                "Take a short break to relax.",
                "Try some deep breathing exercises.",
                "Consider a walk or light activity."
            ],
            2: [  # Moderate stress
                "Try deep breathing exercises: inhale for 4 counts, hold for 4, exhale for 4.",
                "Listen to calming music or nature sounds to relax.",
                "Engage in stress-relief activities like stretching or yoga."
            ],
            3: [  # High stress
                "Try guided meditation to calm your mind.",
                "Write down your thoughts in a journal to process your feelings.",
                "Consider professional support if needed."
            ]
        }

    def get_suggestion(self, stress_level):
        """Get a supportive message based on stress level (0-3)."""
        if stress_level in self.suggestions:
            return np.random.choice(self.suggestions[stress_level])
        else:
            return "Take care of yourself."

# Example usage
if __name__ == "__main__":
    sugg = SupportSuggestions()
    print("Stressed:", sugg.get_suggestion(True))
    print("Not stressed:", sugg.get_suggestion(False))