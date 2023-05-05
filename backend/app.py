import torch
from model import BiLSTM
from preprocess import preprocess_text, load_dataset
from flask import Flask, request, jsonify
from flask_cors import CORS

# Load the dataset to get the vocab object
_, _, _, vocab = load_dataset("dataset.csv")

# Define the model's hyperparameters
vocab_size = len(vocab) 
embedding_dim = 100
hidden_dim = 64
output_dim = 1
num_layers = 2
dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Profanity treshold 
threshold = 0.5

app = Flask(__name__)
CORS(app)

# Load the trained model
model = BiLSTM(vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout)
model.load_state_dict(torch.load("profanity_model.pt"))
model.eval()

@app.route("/check", methods=["POST"])
def predict():
    if not request.json or "text" not in request.json:
        return jsonify({"error": "Missing 'text' key in the JSON request object."}), 400
    
    text = request.json["text"]
    preprocessed_text = preprocess_text(text,vocab)
    tensor_text = torch.tensor([preprocessed_text]).to(device)
    output = model(tensor_text).squeeze(1)
    prob = torch.sigmoid(output).item()

    contains_profanity = prob > threshold

    return jsonify({"profanity_probability": prob, "contains_profanity": contains_profanity})

if __name__ == "__main__":
    app.run(debug=True)