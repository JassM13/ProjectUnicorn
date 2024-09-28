import json

class TrustSystem:
    def __init__(self, trust_file='trust_system.json'):
        self.trust_file = trust_file
        self.user_trust = {}
        self.load_trust_data()

    def load_trust_data(self):
        """Load trust data from a JSON file."""
        try:
            with open(self.trust_file, 'r') as f:
                self.user_trust = json.load(f)
        except FileNotFoundError:
            print("No existing trust data found. Starting fresh.")
            self.user_trust = {}

    def save_trust_data(self):
        """Save trust data to a JSON file."""
        with open(self.trust_file, 'w') as f:
            json.dump(self.user_trust, f, indent=4)

    def update_trust(self, user_id, is_honest=True):
        """Update the trust level of a user."""
        if user_id not in self.user_trust:
            self.user_trust[user_id] = {'trust_score': 0}

        if is_honest:
            self.user_trust[user_id]['trust_score'] += 1
        else:
            self.user_trust[user_id]['trust_score'] -= 1

        # Ensure trust_score remains within bounds
        self.user_trust[user_id]['trust_score'] = max(
            min(self.user_trust[user_id]['trust_score'], 10), -10
        )
        self.save_trust_data()

    def is_trusted(self, user_id):
        """Determine if a user is trusted based on their trust score."""
        return self.user_trust.get(user_id, {'trust_score': 0})['trust_score'] >= 5