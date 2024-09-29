import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import random
import os
import pickle
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from contextlib import contextmanager

# Replace logger with print statements
def log_info(message):
    print(f"INFO: {message}")

def log_error(message):
    print(f"ERROR: {message}")

# Use a simple tokenizer without requiring NLTK downloads
def improved_tokenize(text):
    return text.lower().split()

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=10000, tokenizer=improved_tokenize)

# Improved UnicornAgent architecture
class UnicornAgent(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, dropout=0.3):
        super(UnicornAgent, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_size2, output_size)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# Update preprocessing function
def preprocess(text):
    if pd.isna(text) or not isinstance(text, str):
        return torch.zeros(10000, dtype=torch.float32)
    tfidf_embedding = vectorizer.transform([text]).toarray()[0]
    return torch.tensor(tfidf_embedding, dtype=torch.float32)

# Update ConversationDataset class
class ConversationDataset(Dataset):
    def __init__(self, df):
        self.df = df
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        user_message, assistant_message = self.extract_messages(row)
        
        input_embedding = preprocess(user_message)
        output_embedding = preprocess(assistant_message)
        
        return input_embedding, output_embedding
    
    def extract_messages(self, row):
        messages = row.get('messages', np.nan)
        question = row.get('question', "")
        answer = row.get('answer', "")
        
        user_message = "Hello!"
        assistant_message = "Hello! How can I help you today?"
        
        if isinstance(messages, (list, np.ndarray)):
            messages = messages if isinstance(messages, list) else messages.tolist()
            for message in messages:
                if isinstance(message, dict):
                    role = message.get('role', '').lower()
                    content = message.get('content', '')
                    if role == 'user':
                        user_message = content
                    elif role == 'assistant' and user_message:
                        assistant_message = content
                        break
        elif isinstance(messages, str):
            try:
                parsed_messages = ast.literal_eval(messages)
                if isinstance(parsed_messages, list):
                    for message in parsed_messages:
                        if isinstance(message, dict):
                            role = message.get('role', '').lower()
                            content = message.get('content', '')
                            if role == 'user':
                                user_message = content
                            elif role == 'assistant' and user_message:
                                assistant_message = content
                                break
            except (ValueError, SyntaxError):
                pass
        elif pd.isna(messages):
            if isinstance(question, str) and isinstance(answer, str) and question.strip() and answer.strip():
                user_message = question
                assistant_message = answer
        
        return user_message, assistant_message

@contextmanager
def eval_mode(model):
    """Context manager to set the model to eval mode temporarily."""
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        if was_training:
            model.train()

# Singleton Pattern for UnicornAgent
class AdvancedNLPAgent:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AdvancedNLPAgent, cls).__new__(cls)
            cls._instance.__init__()
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            # Define hyperparameters
            self.input_size = 10000
            self.hidden_size1 = 2048
            self.hidden_size2 = 1024
            self.output_size = 10000
            self.learning_rate = 0.001
            self.batch_size = 64
            self.epochs = 20
            self.selected_model_name = 'UnicornAgent'

            # Ensure consistent model identifier
            self.model_identifier = f"{self.selected_model_name}_input{self.input_size}_hidden1{self.hidden_size1}_hidden2{self.hidden_size2}_output{self.output_size}"

            self.model = UnicornAgent(self.input_size, self.hidden_size1, self.hidden_size2, self.output_size)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            self.criterion = nn.MSELoss()
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=3)
            
            # Load or build the response cache
            self.response_embeddings = load_response_cache(df_combined)
            
            # Check for cached model and train if not found
            cache_available = load_cache(self.model, self.optimizer, vectorizer, self.model_identifier)
            if not cache_available:
                self.train(self.epochs)
                save_cache(self.model, self.optimizer, vectorizer, self.model_identifier)
            
            self.initialized = True

    def train(self, epochs=20):
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_inputs, batch_outputs in dataloader:
                self.optimizer.zero_grad()
                predictions = self.model(batch_inputs)
                loss = self.criterion(predictions, batch_outputs)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(dataloader)
            self.scheduler.step(avg_loss)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    def process_input(self, user_input):
        input_embedding = preprocess(user_input)
        with eval_mode(self.model):
            with torch.no_grad():
                output_embedding = self.model(input_embedding.unsqueeze(0))
        response = get_response(output_embedding.squeeze(), self.response_embeddings)
        return response

# Caching functions
def save_cache(model, optimizer, vectorizer, model_identifier):
    filename = f'cache/{model_identifier}_model_cache.pkl'
    os.makedirs('cache', exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'vectorizer_vocab': vectorizer.vocabulary_
        }, f)
    print(f"Cache saved for {model_identifier}.")

def load_cache(model, optimizer, vectorizer, model_identifier):
    filename = f'cache/{model_identifier}_model_cache.pkl'
    if os.path.exists(filename):
        try:
            with open(filename, 'rb') as f:
                cache = pickle.load(f)
                model.load_state_dict(cache['model_state_dict'])
                optimizer.load_state_dict(cache['optimizer_state_dict'])
                vectorizer.vocabulary_ = cache['vectorizer_vocab']
            model.train()
            print(f"Loaded cached state for {model_identifier}.")
            return True
        except RuntimeError as e:
            print(f"RuntimeError while loading cache for {model_identifier}: {e}")
            print(f"Deleting corrupted cache file: {filename}")
            os.remove(filename)
            return False
        except Exception as e:
            print(f"Error loading cache for {model_identifier}: {e}")
            return False
    else:
        print(f"No cache found for {model_identifier}. Training a new model...")
        return False

# Response cache functions
response_cache_file = 'cache/response_embeddings.pkl'

def build_response_cache(df):
    responses = set()
    for idx, row in df.iterrows():
        messages = row.get('messages', np.nan)
        
        if isinstance(messages, (list, np.ndarray)):
            messages = messages if isinstance(messages, list) else messages.tolist()
            for message in messages:
                if isinstance(message, dict):
                    role = message.get('role', '').lower()
                    content = message.get('content', '')
                    if role == 'assistant' and isinstance(content, str) and content.strip():
                        responses.add(content)
        else:
            if pd.isna(messages):
                question = row.get('question', "")
                answer = row.get('answer', "")
                if pd.isna(question) or pd.isna(answer):
                    continue
                responses.add(answer)
            else:
                if isinstance(messages, str):
                    try:
                        messages = ast.literal_eval(messages)
                    except (ValueError, SyntaxError):
                        messages = []
                
                if isinstance(messages, list):
                    for message in messages:
                        if isinstance(message, dict):
                            role = message.get('role', '').lower()
                            content = message.get('content', '')
                            if role == 'assistant' and isinstance(content, str) and content.strip():
                                responses.add(content)
    response_embeddings = {response: preprocess(response) for response in responses}
    return response_embeddings

def save_response_cache(response_embeddings):
    os.makedirs('cache', exist_ok=True)
    with open(response_cache_file, 'wb') as f:
        pickle.dump(response_embeddings, f)
    print("Response embeddings cached.")

def load_response_cache(df_combined=None):
    if os.path.exists(response_cache_file):
        try:
            with open(response_cache_file, 'rb') as f:
                response_embeddings = pickle.load(f)
            print("Loaded cached response embeddings.")
            return response_embeddings
        except Exception as e:
            print(f"Error loading response cache: {e}")
            if df_combined is not None:
                print("Rebuilding response embeddings cache...")
                response_embeddings = build_response_cache(df_combined)
                save_response_cache(response_embeddings)
                return response_embeddings
            else:
                raise ValueError("df_combined must be provided to build response cache.")
    else:
        if df_combined is None:
            raise ValueError("df_combined must be provided to build response cache.")
        print("Building response embeddings cache...")
        response_embeddings = build_response_cache(df_combined)
        save_response_cache(response_embeddings)
        return response_embeddings

# Function to get the most similar response with Top-K selection
def get_response(output_embedding, response_embeddings, top_k=5):
    similarities = {}
    for response, embedding in response_embeddings.items():
        if embedding.numel() == output_embedding.numel():
            similarity = F.cosine_similarity(output_embedding.unsqueeze(0), embedding.unsqueeze(0))
            similarities[response] = similarity.item()
        else:
            print(f"Skipping response due to size mismatch: {response}")
    
    sorted_responses = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
    top_k = min(int(top_k), len(sorted_responses))
    top_responses = [resp for resp, sim in sorted_responses[:top_k]]
    
    if top_responses:
        return random.choice(top_responses)
    else:
        return "I'm not sure how to respond to that."

# Define hyperparameters
input_size = 10000
hidden_size1 = 2048
hidden_size2 = 1024
output_size = 10000
learning_rate = 0.001
batch_size = 64
epochs = 20

# Select a model
selected_model_name = 'UnicornAgent'
model_identifier = f"{selected_model_name}_input{input_size}_hidden1{hidden_size1}_hidden2{hidden_size2}_output{output_size}"
model = UnicornAgent(input_size, hidden_size1, hidden_size2, output_size)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# Load data from Parquet and CSV files
print("Loading datasets...")

splits = {
    'train_sft': 'data/train_sft-00000-of-00001.parquet',
    'test_sft': 'data/test_sft-00000-of-00001.parquet'
}

try:
    print("Attempting to load train_sft dataset...")
    df1 = pd.read_parquet("hf://datasets/HuggingFaceTB/everyday-conversations-llama3.1-2k/" + splits["train_sft"])
    print("train_sft dataset loaded successfully.")
except Exception as e:
    print(f"Error loading train_sft dataset: {e}")
    df1 = pd.DataFrame()

try:
    print("Attempting to load casual-conversation dataset...")
    df2 = pd.read_csv("hf://datasets/SohamGhadge/casual-conversation/dialog.zip")
    print("casual-conversation dataset loaded successfully.")
except Exception as e:
    print(f"Error loading casual-conversation dataset: {e}")
    df2 = pd.DataFrame()

df_combined = pd.concat([df1, df2], ignore_index=True)

if df_combined.empty:
    print("ERROR: Combined DataFrame is empty. Check dataset loading.")
    raise ValueError("No data available for training.")

print(f"Combined DataFrame shape: {df_combined.shape}")
print(f"Combined DataFrame columns: {df_combined.columns}")
print("First few rows of the Combined DataFrame:")
print(df_combined.head().to_string())

# Fit the TF-IDF vectorizer on all messages
print("Fitting the TF-IDF vectorizer...")
all_messages = (
    df_combined['messages'].dropna().tolist() +
    df_combined['question'].dropna().tolist() +
    df_combined['answer'].dropna().tolist()
)
processed_messages = []

for message in all_messages:
    if isinstance(message, str):
        try:
            messages = ast.literal_eval(message)
            if isinstance(messages, list):
                for msg in messages:
                    content = msg.get('content', '')
                    if isinstance(content, str):
                        processed_messages.append(content)
        except (ValueError, SyntaxError):
            processed_messages.append(message)
    elif isinstance(message, list):
        for msg in message:
            if isinstance(msg, dict):
                content = msg.get('content', '')
                if isinstance(content, str):
                    processed_messages.append(content)
    elif isinstance(message, float):
        continue
    else:
        processed_messages.append(str(message))

print(f"Number of processed messages: {len(processed_messages)}")
print("Sample of processed messages:")
print(processed_messages[:5])

vectorizer.fit(processed_messages)
print("TF-IDF vectorizer fitting complete.")
print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

# Create dataset and dataloader
print("Creating dataset and dataloader...")
dataset = ConversationDataset(df_combined)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print("Dataset and dataloader ready.")

# Load or build the response cache
response_embeddings = load_response_cache(df_combined)

# Initialize the singleton AI agent
agent = AdvancedNLPAgent()

# Save the updated model on exit
def save_on_exit():
    save_cache(agent.model, agent.optimizer, vectorizer, agent.model_identifier)
    with open(response_cache_file, 'wb') as f:
        pickle.dump(response_embeddings, f)
    print("Thank you for using the chatbot! Model and caches have been updated and saved.")

import atexit
atexit.register(save_on_exit)