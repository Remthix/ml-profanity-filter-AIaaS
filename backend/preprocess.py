import pandas as pd
from sklearn.model_selection import train_test_split
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torch
import spacy
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, data, tokenizer, vocab):
        self.data = data
        self.tokenizer = tokenizer
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data.iloc[idx]
        tokens = self.tokenizer(text)
        token_indices = [self.vocab[token] for token in tokens]
        return torch.tensor(token_indices), torch.tensor(label)

def yield_tokens(data_iter):
    tokenizer = get_tokenizer("spacy") #en_core_web_sm
    for _, row in data_iter:
        yield tokenizer(row['text'])

nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"])

def preprocess_text(text, vocab):  # Add vocab as an argument
    tokenized_text = [token.text for token in nlp(text)]
    numerized_text = [vocab[token] for token in tokenized_text]
    return numerized_text

def load_dataset(file_path):
    data = pd.read_csv(file_path)
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    tokenizer = get_tokenizer("spacy")
    vocab = build_vocab_from_iterator(yield_tokens(train_data.iterrows()))

    train_dataset = TextDataset(train_data, tokenizer, vocab)
    test_dataset = TextDataset(test_data, tokenizer, vocab)

    return train_dataset, test_dataset, len(vocab), vocab

if __name__ == "__main__":
    train, test, vocab_size, vocab = load_dataset("dataset.csv")
    print(f"Vocabulary size: {vocab_size}")