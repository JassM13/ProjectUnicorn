from datasets import load_dataset as hf_load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def normalize_column_names(df):
    """Normalize column names to a consistent format."""
    column_mapping = {
        'text': ['text', 'input', 'question', 'prompt', 'context'],
        'response': ['response', 'output', 'answer', 'completion'],
        'label': ['label', 'score', 'rating', 'sentiment']
    }

    # Create a mapping from current columns to normalized columns
    new_columns = {}
    for norm_col, synonyms in column_mapping.items():
        for syn in synonyms:
            if syn in df.columns:
                new_columns[syn] = norm_col
                break  # Use the first matching synonym

    # Rename columns
    df.rename(columns=new_columns, inplace=True)

    # Ensure all required columns are present
    for norm_col in ['text', 'response', 'label']:
        if norm_col not in df.columns:
            if norm_col == 'label':
                df['label'] = 1  # Default label
                logging.warning(f"Missing column '{norm_col}'. Setting default value to 1.")
            else:
                df[norm_col] = ""  # Empty string for missing text/response
                logging.warning(f"Missing column '{norm_col}'. Setting default value to empty string.")

    return df[['text', 'response', 'label']]

def load_and_process_dataset(dataset_name, config_name=None):
    """Load a dataset from Hugging Face and process it."""
    try:
        dataset = hf_load_dataset(dataset_name, config_name)
        available_splits = list(dataset.keys())

        # Dynamically choose an available split
        split = None
        for possible_split in ['train', 'train_sft', 'default', 'validation', 'test']:
            if possible_split in available_splits:
                split = possible_split
                break

        if split is None:
            logging.warning(f"No standard split found for dataset '{dataset_name}'. Using 'train' split by default.")
            split = 'train'

        df = pd.DataFrame(dataset[split])

        # Normalize column names
        df = normalize_column_names(df)

        # Ensure text and response columns are strings
        df['text'] = df['text'].astype(str)
        df['response'] = df['response'].astype(str)

        logging.info(f"Processed dataset '{dataset_name}' with {len(df)} samples.")
        return df
    except Exception as e:
        logging.error(f"Error loading dataset {dataset_name}: {e}")
        return None

def load_datasets(dataset_list):
    """Load and combine multiple datasets."""
    combined_df = pd.DataFrame()
    for i, dataset_info in enumerate(dataset_list):
        if isinstance(dataset_info, str):
            dataset_name = dataset_info
            config_name = None
        elif isinstance(dataset_info, dict):
            dataset_name = dataset_info['name']
            config_name = dataset_info.get('config')
        else:
            logging.error(f"Invalid dataset info at index {i}: {dataset_info}")
            continue

        df = load_and_process_dataset(dataset_name, config_name)
        if df is not None and not df.empty:
            combined_df = pd.concat([combined_df, df], ignore_index=True)
            logging.info(f"Loaded and combined dataset '{dataset_name}'. Total samples: {len(combined_df)}")
        else:
            logging.warning(f"Dataset '{dataset_name}' is empty or failed to load.")

    return combined_df

def load_and_split_dataset(test_size=0.2):
    """Load datasets, combine them, and split into train and test."""
    datasets_to_load = [
        'HuggingFaceTB/everyday-conversations-llama3.1-2k',
        'SohamGhadge/casual-conversation'
    ]

    combined_df = load_datasets(datasets_to_load)

    if combined_df.empty:
        raise ValueError("No data loaded. Check dataset names and accessibility.")

    train_data, test_data = train_test_split(combined_df, test_size=test_size, random_state=42)
    logging.info(f"Split data into {len(train_data)} training samples and {len(test_data)} testing samples.")
    return train_data

def load_new_data():
    """Load new data for online learning."""
    # This function should load new data for online learning
    # For demonstration, we'll load a small subset of a dataset
    new_data = load_and_process_dataset('HuggingFaceTB/everyday-conversations-llama3.1-2k')
    if new_data is not None and not new_data.empty:
        sample_size = min(100, len(new_data))
        sampled_data = new_data.sample(n=sample_size)
        logging.info(f"Loaded {sample_size} new samples for online learning.")
        return sampled_data
    else:
        logging.warning("No new data loaded for online learning.")
        return pd.DataFrame(columns=['text', 'response', 'label'])