import torch
from model import Encoder, Decoder

# Load the saved model and vocab
checkpoint = torch.load("seq2seq_model.pt")
word2idx = checkpoint["word2idx"]
idx2word = checkpoint["idx2word"]

INPUT_DIM = OUTPUT_DIM = len(word2idx)
EMB_DIM = 64
HID_DIM = 128

encoder = Encoder(INPUT_DIM, EMB_DIM, HID_DIM)
decoder = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM)

encoder.load_state_dict(checkpoint["encoder_state"])
decoder.load_state_dict(checkpoint["decoder_state"])

encoder.eval()
decoder.eval()

# Utilities
def tokenize(sentence):
    return [word2idx.get(word, word2idx["<pad>"]) for word in sentence.lower().split()] + [word2idx["<eos>"]]

def detokenize(indices):
    words = []
    for idx in indices:
        if idx2word[idx] == "<eos>":
            break
        words.append(idx2word.get(idx, "<?>"))
    return " ".join(words)

# Predict Function
def generate_reply(input_text):
    src = torch.tensor(tokenize(input_text)).unsqueeze(1)  # [seq_len, 1]
    encoder_hidden = encoder(src)

    input_token = torch.tensor([word2idx["<sos>"]])  # start with <sos>
    generated = []

    for _ in range(20):  # max output length
        output, encoder_hidden = decoder(input_token, encoder_hidden)
        pred_token = output.argmax(1)
        if pred_token.item() == word2idx["<eos>"]:
            break
        generated.append(pred_token.item())
        input_token = pred_token

    return detokenize(generated)

# 🔁 Test it with input
while True:
    user_input = input("You: ")
    if user_input.lower() in ['quit', 'exit']:
        break
    reply = generate_reply(user_input)
    print("Bot:", reply)
