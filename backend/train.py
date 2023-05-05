import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from torch.utils.data import DataLoader
from model import BiLSTM
from preprocess import load_dataset

def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0

    for batch in iterator:
        text, label = batch
        text, label = text.to(device), label.to(device)
        optimizer.zero_grad()
        predictions = model(text).squeeze(1)
        loss = criterion(predictions, label.float())
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def custom_collate(batch):
    texts, labels = zip(*batch)
    padded_texts = pad_sequence([torch.tensor(t) for t in texts], batch_first=True)
    labels = torch.tensor(labels)
    return padded_texts, labels

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data, test_data, vocab_size, vocab = load_dataset("dataset.csv")

    batch_size = 32
    train_iterator = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    test_iterator = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)

    embedding_dim = 100
    hidden_dim = 64
    output_dim = 1
    num_layers = 2
    dropout = 0.5

    model = BiLSTM(vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout).to(device)

    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()

    num_epochs = 10

    for epoch in range(num_epochs):
        train_loss = train(model, train_iterator, optimizer, criterion)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}")

    torch.save(model.state_dict(), "profanity_model.pt")