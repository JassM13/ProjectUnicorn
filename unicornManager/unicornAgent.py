import numpy as np
import json
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import faiss  # Make sure to install faiss-cpu
import datetime

from unicornManager.trustSystem import TrustSystem
from unicornManager.factChecker import FactChecker
from unicornManager.memoryNode import MemoryNode

class VectorDatabase:
    def __init__(self, filename='vector_database.json'):
        self.vectorizer = TfidfVectorizer()
        self.data = {}
        self.vectors = None
        self.index = None
        self.filename = filename
        self.load_data()

    def add_document(self, input_text, response):
        """Add a new input and its response to the database if it's not a duplicate."""
        # Check for similarity with existing documents
        existing_inputs = list(self.data.keys())
        if existing_inputs:
            similarities = cosine_similarity(
                self.vectorizer.transform([input_text]).toarray(),
                self.vectorizer.transform(existing_inputs).toarray()
            )
            max_similarity = similarities.max()
            if max_similarity > 0.9:
                print("Similar input already exists. Skipping addition to prevent duplication.")
                return
        
        if input_text not in self.data:
            self.data[input_text] = []
        self.data[input_text].append(response)
        self.update_vectors()

    def update_vectors(self):
        """Update the vectors and index for the documents."""
        self.vectors = self.vectorizer.fit_transform(list(self.data.keys())).toarray()
        self.index = faiss.IndexFlatL2(self.vectors.shape[1])  # L2 distance
        self.index.add(np.array(self.vectors).astype('float32'))  # Convert to float32 for FAISS

    def load_data(self):
        """Load documents from a local JSON file."""
        try:
            with open(self.filename, 'r') as f:
                self.data = json.load(f)
                self.update_vectors()
        except FileNotFoundError:
            print("No existing vector database found. Starting fresh.")

    def save_data(self):
        """Save the current documents to a local JSON file."""
        with open(self.filename, 'w') as f:
            json.dump(self.data, f, indent=4)

    def search(self, query, k=1):
        """Search for similar documents based on a query."""
        query_vector = self.vectorizer.transform([query]).toarray().astype('float32')
        distances, indices = self.index.search(query_vector, k)  # k nearest neighbors
        return [(list(self.data.keys())[idx], distances[0][i]) for i, idx in enumerate(indices[0])] if self.index and self.data else []

class AdvancedNLPAgent:
    _instance = None

    def __new__(cls, user_id):
        """Ensure that only one instance of AdvancedNLPAgent is created per user."""
        if cls._instance is None:
            cls._instance = super(AdvancedNLPAgent, cls).__new__(cls)
            cls._instance.vector_db = VectorDatabase()
            cls._instance.trust_system = TrustSystem()
            cls._instance.fact_checker = FactChecker()
            cls._instance.user_id = user_id  # Associate agent with a user
            cls._instance.__init_memory()
        return cls._instance

    def __init_memory(self):
        """Initialize memory nodes based on the vector database."""
        self.memory_nodes = {}
        for input_text in self.vector_db.data:
            self.memory_nodes[input_text] = MemoryNode(input_text)

    def add_memory_connection(self, from_input, to_input, relevance=1):
        """Connect two memory nodes."""
        if from_input in self.memory_nodes and to_input in self.memory_nodes:
            self.memory_nodes[from_input].add_connection(to_input, relevance)

    def get_related_memories(self, input_text):
        """Retrieve related memories based on connections."""
        node = self.memory_nodes.get(input_text)
        if node and node.is_recent():
            return node.connections.keys()
        return []

    def is_factual_input(self, user_input):
        """Determine if the input is a factual statement requiring verification."""
        # Define criteria for factual statements. This can be enhanced with NLP techniques.
        factual_keywords = ['is', 'are', 'do', 'does', 'did', 'can', 'could', 'should', 'would', 'have', 'has', 'had']
        return any(word in user_input.lower().split() for word in factual_keywords)

    def verify_user_input(self, user_input):
        """Verify the user input's trustworthiness using the FactChecker."""
        fact_check_result = self.fact_checker.verify(user_input)
        if fact_check_result.get('is_true', False):
            self.trust_system.update_trust(self.user_id, is_honest=True)
            return True
        else:
            self.trust_system.update_trust(self.user_id, is_honest=False)
            return False

    def generate_response(self, input_text):
        """Generate a response based on the input text with reasoning."""
        if self.is_factual_input(input_text):
            # Handle factual input with verification
            if self.verify_user_input(input_text):
                results = self.vector_db.search(input_text, k=1)
                if results:
                    document, _ = results[0]
                    responses = self.vector_db.data[document]
                    chosen_response = random.choice(responses)
                    reasoning = self.reasoning(chosen_response, input_text)
                    return f"{chosen_response}\n*Reasoning:* {reasoning}"
                else:
                    # Unknown factual input; ask the user for verification
                    return "I encountered something new. Could you please provide more details or verify the information?"
            else:
                # Low trust input; challenge the user
                return "I'm uncertain about that information. Can you provide credible sources or further explanation?"
        else:
            # Handle normal conversational input without verification
            results = self.vector_db.search(input_text, k=1)
            if results:
                document, _ = results[0]
                responses = self.vector_db.data[document]
                chosen_response = random.choice(responses)
                reasoning = self.reasoning(chosen_response, input_text)
                return f"{chosen_response}\n*Reasoning:* {reasoning}"
            
            # If no information is found, ask the user for the answer without invoking Trust System
            return "I'm not sure about that. Can you please tell me how to respond?"

    def reasoning_engine(self, input_text, response):
        """Provide enhanced reasoning behind the response."""
        # Implement a simple rule-based reasoning example
        if "Greetings" in response or "Hi" in response or "Hello" in response:
            return "The response is a standard greeting appropriate to the input."
        elif "assist you" in response or "help" in response:
            return "Offering assistance aligns with the user's inquiry about help."
        else:
            return "The response is generated based on the closest matching input."

    def reasoning(self, response, input_text):
        """Generate reasoning for the response."""
        return self.reasoning_engine(input_text, response)

    def learn(self, user_input, user_response):
        """Add a new phrase and response to the agent's knowledge and manage memory connections."""
        self.vector_db.add_document(user_input, user_response)
        self.vector_db.save_data()
        # Connect the new input to existing inputs if they share keywords
        keywords = user_input.split()
        for existing_input in self.vector_db.data:
            existing_keywords = existing_input.split()
            common = set(keywords).intersection(set(existing_keywords))
            if common and existing_input != user_input:
                self.add_memory_connection(user_input, existing_input)
        # Update memory timestamp
        if user_input in self.memory_nodes:
            self.memory_nodes[user_input].update_timestamp()

    def process_input(self, user_input):
        """Process user input, generate a response, and handle dynamic learning."""
        response = self.generate_response(user_input)

        # If the response asks for more information, handle learning
        if "Could you please provide more details" in response or "Can you provide credible sources" in response:
            user_provided_response = input("Please provide a verified response for this input: ")
            if user_provided_response:
                # Only update trust if the user provides a trustworthy response
                is_trusted = self.trust_system.is_trusted(self.user_id)
                if is_trusted:
                    self.learn(user_input, user_provided_response)
                    return "Thanks! I've learned that response."
                else:
                    return "The information provided doesn't meet trust criteria. Please provide credible sources."
            else:
                return "No response provided. Cannot learn."

        return response

    def interact(self, user_input):
        """Simulate interaction with the user."""
        response = self.process_input(user_input)
        print(response)