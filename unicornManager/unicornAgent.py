import argparse
from .NLPAgent import GenerativeNLPAgent
from .data_loader import load_and_split_dataset, load_new_data
import pandas as pd
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

agent = GenerativeNLPAgent()

def main():
    parser = argparse.ArgumentParser(description="Generative NLP Agent")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--chat", action="store_true", help="Start chat session")
    args = parser.parse_args()

    if args.train:
        logging.info("Loading dataset for training...")
        try:
            train_data = load_and_split_dataset()
            logging.info(f"Loaded {len(train_data)} training samples.")
            logging.info("Training the model...")
            agent.train(train_data, epochs=1, batch_size=64, teacher_force_ratio=0.5)  # Reduced epochs for quick testing
            logging.info("Initial training complete.")
        except Exception as e:
            logging.error(f"Error during training: {e}")

    if args.chat:
        logging.info("Starting chat session. Type 'quit' to exit.")
        print("The AI is learning in the background. It may take a moment to start responding.")
        chat_history = []
        while True:
            try:
                user_input = input("You: ")
            except (EOFError, KeyboardInterrupt):
                print("\nExiting chat session.")
                logging.info("Chat session terminated by user.")
                break
            if user_input.lower() in ['train']:
                logging.info("Loading dataset for training...")
                try:
                    train_data = load_and_split_dataset()
                    logging.info(f"Loaded {len(train_data)} training samples.")
                    logging.info("Training the model...")
                    agent.train(train_data, epochs=1, batch_size=64, teacher_force_ratio=0.5)  # Reduced epochs for quick testing
                    logging.info("Initial training complete.")
                except Exception as e:
                    logging.error(f"Error during training: {e}")
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Exiting chat session.")
                logging.info("Chat session terminated by user typing quit/exit/bye.")
                break
            response = agent.generate_response(user_input)
            print(f"AI: {response}")
            logging.info(f"You: {user_input} | AI: {response}")

            # Collect chat data for online learning
            chat_history.append({
                'text': user_input,
                'response': response,
                'label': 1  # Assuming positive label for all responses
            })

            # Perform online learning every 5 interactions
            if len(chat_history) % 5 == 0:
                chat_data = pd.DataFrame(chat_history[-5:])  # Last 5 interactions
                agent.online_learn(chat_data)
                logging.info("Performed online learning with recent chat data.")

    # If not chatting or training, start continuous online learning simulation
    if not args.chat and not args.train:
        logging.info("Starting continuous online learning simulation...")
        print("Starting continuous online learning simulation...")
        try:
            while True:
                new_data = load_new_data()
                logging.info(f"Loaded {len(new_data)} new samples for online learning.")
                agent.online_learn(new_data)
                logging.info("Submitted new data for background learning.")
                time.sleep(60)  # Reduced sleep time for quicker testing
        except KeyboardInterrupt:
            logging.info("Continuous learning simulation stopped by user.")
            print("Continuous learning simulation stopped.")
        except Exception as e:
            logging.error(f"Error during continuous learning: {e}")

if __name__ == "__main__":
    main()