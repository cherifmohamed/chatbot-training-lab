import torch
import torch.nn as nn
import torch.optim as optim
from model import TextGenGRU
from utils import tokenize
import os
import json

# === Load and prepare data ===
with open("data/dialog.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

# === Split into parts (1/100 each run) ===
PARTS = 100
lines_per_part = len(lines) // PARTS

# === Training Loop over all parts ===
for part_index in range(PARTS):
    start = part_index * lines_per_part
    end = (part_index + 1) * lines_per_part if part_index < PARTS - 1 else len(lines)
    part_lines = lines[start:end]

    text = " ".join([line.strip() for line in part_lines if line.strip()])
    tokens = text.lower().split()

    special_tokens = ["<pad>", "<eos>", "<sos>"]
    words = sorted(set(tokens) | set(special_tokens))
    word2idx = {w: i for i, w in enumerate(words)}
    idx2word = {i: w for w, i in word2idx.items()}

    vocab_size = len(word2idx)
    print(f"📚 Vocab size: {vocab_size}")
    print(f"🧪 Training part {part_index+1}/{PARTS} on {len(part_lines)} lines")

    encoded = [word2idx.get(word, word2idx["<pad>"]) for word in tokens]
    input_seq = torch.tensor(encoded[:-1]).unsqueeze(1)
    target_seq = torch.tensor(encoded[1:]).unsqueeze(1)

    EMB_DIM = 64
    HID_DIM = 128
    N_EPOCHS = 10
    LR = 0.005

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TextGenGRU(vocab_size, EMB_DIM, HID_DIM).to(device)
    model_path = "models/textgen_gru.pt"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    print("🚀 Starting training...")
    for epoch in range(N_EPOCHS):
        model.train()
        optimizer.zero_grad()

        output, _ = model(input_seq.to(device))
        output = output.view(-1, vocab_size)
        loss = criterion(output, target_seq.view(-1).to(device))

        loss.backward()
        optimizer.step()

        print(f"Part {part_index+1}/{PARTS} | Epoch {epoch+1}/{N_EPOCHS} | Loss: {loss.item():.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), model_path)
    with open("models/vocab.json", "w", encoding="utf-8") as f:
        json.dump({"word2idx": word2idx, "idx2word": idx2word}, f, indent=2)

    print("✅ Model and vocab saved for part.")
