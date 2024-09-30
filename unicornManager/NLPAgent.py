import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import pickle
import threading
import queue
import time
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# Special tokens
PAD_TOKEN = '<PAD>'
SOS_TOKEN = '<SOS>'
EOS_TOKEN = '<EOS>'
UNK_TOKEN = '<UNK>'

# Define Vocabulary class
class Vocabulary:
    def __init__(self):
        self.word2index = {}
        self.index2word = {}
        self.n_words = 0  # Count of words

        # Initialize with special tokens
        self.add_word(PAD_TOKEN)
        self.add_word(SOS_TOKEN)
        self.add_word(EOS_TOKEN)
        self.add_word(UNK_TOKEN)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1

# Define the dataset
class ConversationDataset(Dataset):
    def __init__(self, data, input_vocab, output_vocab, max_length=50):
        self.data = data
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            input_sentence = self.data.iloc[idx]['text']
            target_sentence = self.data.iloc[idx]['response']
        except IndexError as e:
            logging.error(f"IndexError at idx {idx}: {e}")
            raise e

        # Handle missing or empty texts
        input_sentence = input_sentence if isinstance(input_sentence, str) else ""
        target_sentence = target_sentence if isinstance(target_sentence, str) else ""

        # Tokenize by splitting on spaces
        input_tokens = input_sentence.lower().split(' ')
        target_tokens = target_sentence.lower().split(' ')

        # Convert tokens to indices
        input_indices = [self.input_vocab.word2index.get(SOS_TOKEN, 0)]
        for word in input_tokens:
            input_indices.append(self.input_vocab.word2index.get(word, self.input_vocab.word2index.get(UNK_TOKEN, 1)))
        input_indices.append(self.input_vocab.word2index.get(EOS_TOKEN, 2))

        target_indices = [self.output_vocab.word2index.get(SOS_TOKEN, 0)]
        for word in target_tokens:
            target_indices.append(self.output_vocab.word2index.get(word, self.output_vocab.word2index.get(UNK_TOKEN, 1)))
        target_indices.append(self.output_vocab.word2index.get(EOS_TOKEN, 2))

        # Truncate if longer than max_length
        input_indices = input_indices[:self.max_length]
        target_indices = target_indices[:self.max_length]

        return torch.tensor(input_indices, dtype=torch.long), torch.tensor(target_indices, dtype=torch.long)

def collate_fn(batch):
    src_batch, trg_batch = zip(*batch)
    src_padded = nn.utils.rnn.pad_sequence(src_batch, padding_value=0, batch_first=True)
    trg_padded = nn.utils.rnn.pad_sequence(trg_batch, padding_value=0, batch_first=True)
    return src_padded, trg_padded

# Define the Encoder
class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, num_layers=1):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_size, embed_size, padding_idx=0)
        self.gru = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x, lengths):
        embedded = self.embedding(x)  # [batch_size, seq_len, embed_size]
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        outputs, hidden = self.gru(packed)
        return hidden

# Define the Decoder
class Decoder(nn.Module):
    def __init__(self, output_size, embed_size, hidden_size, num_layers=1):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_size, embed_size, padding_idx=0)
        self.gru = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        embedded = self.embedding(x).unsqueeze(1)  # [batch_size, 1, embed_size]
        output, hidden = self.gru(embedded, hidden)  # output: [batch_size, 1, hidden_size]
        prediction = self.out(output.squeeze(1))  # [batch_size, output_size]
        return prediction, hidden

# Define Seq2Seq Model
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, src_lengths, trg, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        trg_len = trg.size(1)
        trg_vocab_size = self.decoder.out.out_features

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        hidden = self.encoder(src, src_lengths)

        # First input to the decoder is the <SOS> tokens
        input = trg[:, 0]

        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden)
            outputs[:, t] = output

            top1 = output.argmax(1)

            # Decide whether to do teacher forcing
            teacher_force = np.random.random() < teacher_forcing_ratio
            input = trg[:, t] if teacher_force else top1

        return outputs

# Define the Agent
class GenerativeNLPAgent:
    def __init__(self, embed_size=256, hidden_size=256, num_layers=1, max_length=50):
        self.input_vocab = Vocabulary()
        self.output_vocab = Vocabulary()
        self.max_length = max_length

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize Encoder and Decoder
        self.encoder = Encoder(self.input_vocab.n_words, embed_size, hidden_size, num_layers)
        self.decoder = Decoder(self.output_vocab.n_words, embed_size, hidden_size, num_layers)

        self.model = Seq2Seq(self.encoder, self.decoder, self.device).to(self.device)

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.input_vocab.word2index.get(PAD_TOKEN, 0))
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # Load or initialize model and vocabularies
        self.load_or_initialize()

        # Set up online learning
        self.learning_queue = queue.Queue()
        threading.Thread(target=self.background_learning, daemon=True).start()

    def load_or_initialize(self):
        # Load vocabularies
        if os.path.exists('../cache/input_vocab.pkl') and os.path.exists('../cache/output_vocab.pkl'):
            try:
                with open('../cache/input_vocab.pkl', 'rb') as f:
                    self.input_vocab = pickle.load(f)
                with open('../cache/output_vocab.pkl', 'rb') as f:
                    self.output_vocab = pickle.load(f)
                logging.info("Vocabularies loaded.")
            except (ModuleNotFoundError, ImportError):
                logging.warning("Failed to load existing vocabularies. Initializing new ones.")
                self.input_vocab = Vocabulary()
                self.output_vocab = Vocabulary()
        else:
            logging.info("Vocabularies not found. They will be built during training.")

    # ... rest of the method remains the same ...

        # Adjust Embedding layers based on vocab size
        if self.input_vocab.n_words > 4:
            self.encoder.embedding = nn.Embedding(self.input_vocab.n_words, self.encoder.embedding.embedding_dim, padding_idx=0).to(self.device)
        if self.output_vocab.n_words > 4:
            self.decoder.embedding = nn.Embedding(self.output_vocab.n_words, self.decoder.embedding.embedding_dim, padding_idx=0).to(self.device)
            self.decoder.out = nn.Linear(self.decoder.gru.hidden_size, self.output_vocab.n_words).to(self.device)

        # Load model weights if exists
        if os.path.exists('./cache/seq2seq_model.pth'):
            try:
                self.model.load_state_dict(torch.load('./cache/seq2seq_model.pth', map_location=self.device))
                logging.info("Model loaded from checkpoint.")
            except Exception as e:
                logging.error(f"Error loading model checkpoint: {e}")
                logging.info("Starting with a fresh model.")
        else:
            logging.info("No model checkpoint found. Starting with a fresh model.")

    def build_vocabulary(self, data):
        for _, row in data.iterrows():
            for word in row['text'].lower().split(' '):
                self.input_vocab.add_word(word)
            for word in row['response'].lower().split(' '):
                self.output_vocab.add_word(word)
        self.save_vocab()
        logging.info(f"Input Vocabulary Size: {self.input_vocab.n_words}")
        logging.info(f"Output Vocabulary Size: {self.output_vocab.n_words}")

    def save_vocab(self):
        with open('cache/input_vocab.pkl', 'wb') as f:
            pickle.dump(self.input_vocab, f)
        with open('cache/output_vocab.pkl', 'wb') as f:
            pickle.dump(self.output_vocab, f)
        logging.info("Vocabularies saved.")

    def save_model(self):
        torch.save(self.model.state_dict(), 'cache/seq2seq_model.pth')
        logging.info("Model saved.")

    def train(self, train_data, epochs=1, batch_size=64, teacher_force_ratio=0.5):
        # Build vocabularies if not loaded
        if self.input_vocab.n_words <= 4 or self.output_vocab.n_words <= 4:
            logging.info("Building vocabularies from training data...")
            self.build_vocabulary(train_data)

            # Reinitialize Encoder and Decoder with new vocab sizes
            self.encoder = Encoder(self.input_vocab.n_words, self.encoder.embedding.embedding_dim, self.encoder.gru.hidden_size, self.encoder.gru.num_layers).to(self.device)
            self.decoder = Decoder(self.output_vocab.n_words, self.decoder.embedding.embedding_dim, self.decoder.gru.hidden_size, self.decoder.gru.num_layers).to(self.device)
            self.model = Seq2Seq(self.encoder, self.decoder, self.device).to(self.device)

        # Create dataset and dataloader
        dataset = ConversationDataset(train_data, self.input_vocab, self.output_vocab, max_length=self.max_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            unk_count = 0
            total_tokens = 0
            for batch_idx, (src, trg) in enumerate(dataloader):
                src, trg = src.to(self.device), trg.to(self.device)
                src_lengths = torch.sum(src != self.input_vocab.word2index.get(PAD_TOKEN, 0), dim=1)

                self.optimizer.zero_grad()
                output = self.model(src, src_lengths, trg, teacher_force_ratio)

                # Reshape for loss
                output_dim = output.shape[-1]
                output = output[:, 1:].reshape(-1, output_dim)
                trg = trg[:, 1:].reshape(-1)

                loss = self.criterion(output, trg)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

                # Count <UNK> tokens in targets
                unk_idx = self.output_vocab.word2index.get(UNK_TOKEN, 1)
                unk_tokens = (trg == unk_idx).sum().item()
                unk_count += unk_tokens
                total_tokens += trg.numel()

                if (batch_idx + 1) % 100 == 0:
                    logging.info(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

            avg_loss = epoch_loss / len(dataloader)
            unk_ratio = unk_count / total_tokens if total_tokens > 0 else 0
            logging.info(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}, <UNK> Ratio: {unk_ratio:.4f}")
            self.save_model()

    def generate_response(self, user_input, max_length=50):
        # Check if vocabularies are built
        if self.input_vocab.n_words <= 4 or self.output_vocab.n_words <= 4:
            logging.warning("Vocabularies not loaded. Please train the model first using '--train'.")
            return "I'm not trained yet. Please train me first using the '--train' option."

        self.model.eval()
        with torch.no_grad():
            # Tokenize input
            input_indices = [self.input_vocab.word2index.get(word, self.input_vocab.word2index.get(UNK_TOKEN, 1)) for word in user_input.lower().split(' ')]
            input_tensor = torch.tensor([self.input_vocab.word2index.get(SOS_TOKEN, 0)] + input_indices + [self.input_vocab.word2index.get(EOS_TOKEN, 2)], dtype=torch.long).unsqueeze(0).to(self.device)

            src_lengths = torch.tensor([len(input_indices) + 2]).to(self.device)  # +2 for SOS and EOS

            # Encode input
            try:
                hidden = self.encoder(input_tensor, src_lengths)
            except Exception as e:
                logging.error(f"Error during encoding: {e}")
                return "Sorry, I couldn't process that."

            # Decode
            trg_indexes = [self.output_vocab.word2index.get(SOS_TOKEN, 0)]
            for _ in range(max_length):
                trg_tensor = torch.tensor([trg_indexes[-1]], dtype=torch.long).to(self.device)
                try:
                    output, hidden = self.decoder(trg_tensor, hidden)
                except Exception as e:
                    logging.error(f"Error during decoding: {e}")
                    return "Sorry, I couldn't generate a response."

                pred_token = output.argmax(1).item()
                trg_indexes.append(pred_token)

                if pred_token == self.output_vocab.word2index.get(EOS_TOKEN, 2):
                    break

            # Convert indices to words
            trg_tokens = [self.output_vocab.index2word.get(idx, UNK_TOKEN) for idx in trg_indexes]
            response = ' '.join(trg_tokens[1:-1])  # Exclude SOS and EOS

            if not response:
                return "I'm still learning how to respond to that."

            return response

    def online_learn(self, new_data, epochs=1, batch_size=16, teacher_force_ratio=0.5):
        self.learning_queue.put((new_data, epochs, batch_size, teacher_force_ratio))
        logging.info("New data queued for online learning.")

    def background_learning(self):
        while True:
            try:
                new_data, epochs, batch_size, teacher_force_ratio = self.learning_queue.get(timeout=60)
                if new_data is not None and len(new_data) > 0:
                    logging.info("Starting online learning with new data...")
                    self.train(new_data, epochs=epochs, batch_size=batch_size, teacher_force_ratio=teacher_force_ratio)
                    logging.info("Online learning completed.")
                else:
                    logging.warning("No new data available for learning.")
            except queue.Empty:
                logging.debug("No new data received in the last 60 seconds. Continuing to wait...")
            except Exception as e:
                logging.error(f"Error in background learning: {e}")
            time.sleep(1)