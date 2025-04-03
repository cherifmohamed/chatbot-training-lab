import torch
import torch.nn as nn
from utils import tokenize

class TextGenGRU(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, num_layers=1, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.gru = nn.GRU(emb_dim, hid_dim, num_layers, dropout=dropout if num_layers > 1 else 0)
        self.fc_out = nn.Linear(hid_dim, vocab_size)
        
        # Optional: Weight tying
        if emb_dim == hid_dim:
            self.fc_out.weight = self.embedding.weight

    def forward(self, input_seq, hidden=None):
        embedded = self.embedding(input_seq)  # (seq_len, batch, emb_dim)
        output, hidden = self.gru(embedded, hidden)
        predictions = self.fc_out(output)  # (seq_len, batch, vocab_size)
        return predictions, hidden

    def generate(self, start_prompt, tokenizer, max_len=50, temperature=0.8, top_k=50):
        self.eval()
        device = next(self.parameters()).device
        input_ids = tokenizer.encode(start_prompt)
        input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(1).to(device)  # (seq_len, 1)
        
        # Initialize hidden state
        hidden = torch.zeros(self.gru.num_layers, 1, self.gru.hidden_size).to(device)
        
        # Feed initial input to get hidden state
        with torch.no_grad():
            _, hidden = self.forward(input_tensor, hidden)
        
        # Start generation with the last token
        next_token = input_tensor[-1, :]
        generated = []
        for _ in range(max_len):
            with torch.no_grad():
                output, hidden = self.forward(next_token.unsqueeze(0), hidden)
                logits = output.squeeze(0) / max(temperature, 1e-3)
                
                # Top-k sampling
                if top_k > 0:
                    top_logits, top_indices = logits.topk(top_k)
                    probs = torch.softmax(top_logits, dim=-1)
                    next_token = top_indices[torch.multinomial(probs, 1)]
                else:
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                
                token = tokenizer.decode(next_token.item())
                if token == "<eos>":
                    break
                generated.append(token)
                
            next_token = next_token.to(device)
        
        return " ".join(generated)