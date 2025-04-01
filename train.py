import torch
import torch.nn as nn
import json
import os
import csv
import random
from model import Encoder, Decoder
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# === Loop through all remaining configs ===
while True:
    # === Load config file and find the next unfinished config ===
    with open("train_configs.json", "r") as f:
        configs = json.load(f)

    current_config = None
    for config in configs:
        if not config.get("done", False):
            current_config = config
            break

    if not current_config:
        print("✅ All experiments completed!")
        break

    # Extract parameters
    EMB_DIM = current_config["emb_dim"]
    HID_DIM = current_config["hid_dim"]
    N_EPOCHS = current_config["epochs"]
    BATCH_SIZE = current_config["batch_size"]
    LEARNING_RATE = current_config.get("learning_rate", 0.001)
    TEACHER_FORCING = current_config.get("teacher_forcing", 1.0)

    # === Load training data ===
    with open("data/train_pairs.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    sampled = random.sample(data, 1000)
    inputs = [pair["input"] for pair in sampled]
    outputs = [pair["output"] for pair in sampled]

    print(f"Loaded {len(inputs)} training pairs.")

    # === Build vocabulary ===
    words = set()
    for sentence in inputs + outputs:
        for word in sentence.lower().split():
            words.add(word)

    word2idx = {w: i+2 for i, w in enumerate(words)}
    word2idx["<pad>"] = 0
    word2idx["<sos>"] = 1
    word2idx["<eos>"] = len(word2idx)
    idx2word = {i: w for w, i in word2idx.items()}

    def tokenize(sentence):
        return [word2idx.get(word, 0) for word in sentence.lower().split()] + [word2idx["<eos>"]]

    X = [torch.tensor(tokenize(s)) for s in inputs]
    Y = [torch.tensor([word2idx["<sos>"]] + tokenize(s)) for s in outputs]

    # === Prepare model ===
    INPUT_DIM = OUTPUT_DIM = len(word2idx)
    encoder = Encoder(INPUT_DIM, EMB_DIM, HID_DIM)
    decoder = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=LEARNING_RATE)

    def batch_data(X, Y):
        dataset = list(zip(X, Y))
        return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda batch: (
            pad_sequence([x for x, _ in batch], padding_value=0),
            pad_sequence([y for _, y in batch], padding_value=0)
        ))

    # === Training loop ===
    total_tokens = 0
    correct_tokens = 0
    dataloader = batch_data(X, Y)
    num_batches = len(dataloader)

    for epoch in range(N_EPOCHS):
        epoch_loss = 0

        for src_batch, trg_batch in dataloader:
            encoder_hidden = encoder(src_batch)
            input_token = trg_batch[0]
            loss = 0

            for t in range(1, trg_batch.size(0)):
                output, encoder_hidden = decoder(input_token, encoder_hidden)
                target = trg_batch[t]
                loss += criterion(output, target)

                predictions = output.argmax(1)
                correct_tokens += (predictions == target).sum().item()
                total_tokens += target.size(0)

                input_token = target if torch.rand(1).item() < TEACHER_FORCING else predictions

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{N_EPOCHS} | Loss: {epoch_loss / num_batches:.4f}")

    # === Save model ===
    final_loss = epoch_loss / num_batches
    accuracy = (correct_tokens / total_tokens) * 100

    os.makedirs("models_train", exist_ok=True)
    model_name = f"models_train/{accuracy:.2f}_{final_loss:.2f}_{num_batches}_{EMB_DIM}_{HID_DIM}_{N_EPOCHS}_{BATCH_SIZE}_seq2seq_model.pt"

    torch.save({
        'encoder_state': encoder.state_dict(),
        'decoder_state': decoder.state_dict(),
        'word2idx': word2idx,
        'idx2word': idx2word
    }, model_name)
    print(f"✅ Model saved as {model_name}")

    # === Save results to CSV ===
    os.makedirs("logs", exist_ok=True)
    csv_file = "logs/train_results.csv"
    write_header = not os.path.exists(csv_file)

    with open(csv_file, mode="a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(["accuracy", "loss", "batches", "emb_dim", "hid_dim", "epochs", "batch_size", "learning_rate", "teacher_forcing", "model_name"])
        writer.writerow([f"{accuracy:.2f}", f"{final_loss:.2f}", num_batches, EMB_DIM, HID_DIM, N_EPOCHS, BATCH_SIZE, LEARNING_RATE, TEACHER_FORCING, model_name])

    # === Mark config as done ===
    for config in configs:
        if config == current_config:
            config["done"] = True
            break

    with open("train_configs.json", "w") as f:
        json.dump(configs, f, indent=2)

    print("🔁 Moving to next config... Press Ctrl+C to stop.")
