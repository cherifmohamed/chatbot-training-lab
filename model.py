import torch
import torch.nn as nn
from utils import tokenize

class TextGenGRU(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, num_layers=1):
        super(TextGenGRU, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.gru = nn.GRU(emb_dim, hid_dim, num_layers)
        self.fc_out = nn.Linear(hid_dim, vocab_size)

    def forward(self, input_seq, hidden=None):
        embedded = self.embedding(input_seq)
        outputs, hidden = self.gru(embedded, hidden)
        predictions = self.fc_out(outputs)
        return predictions, hidden

    def generate(self, start_prompt, word2idx, idx2word, max_len=30, temperature=0.8):
        self.eval()
        device = next(self.parameters()).device
        input_ids = tokenize(start_prompt, word2idx)
        input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(1).to(device)

        hidden = None
        _, hidden = self.forward(input_tensor, hidden)
        input_token = input_tensor[-1].unsqueeze(0)

        generated = []
        for _ in range(max_len):
            output, hidden = self.forward(input_token, hidden)
            logits = output[-1, 0] / temperature
            probs = torch.softmax(logits, dim=0)
            next_token = torch.multinomial(probs, 1).item()

            if idx2word.get(next_token) == "<eos>":
                break
            generated.append(idx2word.get(next_token, "<?>"))
            input_token = torch.tensor([[next_token]]).to(device)

        return " ".join(generated)