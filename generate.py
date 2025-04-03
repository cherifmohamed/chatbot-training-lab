import torch
import json
from model import TextGenGRU
from utils import tokenize

with open("models/vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)
    word2idx = vocab["word2idx"]
    idx2word = {int(k): v for k, v in vocab["idx2word"].items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = len(word2idx)
model = TextGenGRU(vocab_size, emb_dim=64, hid_dim=128)
model.load_state_dict(torch.load("models/textgen_gru.pt", map_location=device))
model.to(device)
model.eval()

print("🤖 Chatbot ready. Type 'exit' to quit.")
while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ["exit", "quit"]:
        break
    prompt = f"<sos> user : {user_input} <eos> <sos> bot :"
    reply = model.generate(prompt, word2idx, idx2word)
    print("Bot:", reply)
