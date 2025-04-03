import os
import re
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
import torch

class TextPreprocessor:
    def __init__(self, vocab_size=10000, seq_length=50):
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.word2idx = {"<pad>": 0, "<unk>": 1, "<eos>": 2}
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        
    def get_wiki_text(self):
        """Download Wikipedia dataset using Hugging Face datasets"""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install datasets: pip install datasets")
            
        print("Downloading dataset...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        return "\n".join(dataset["train"]["text"])

    def clean_text(self, text):
        """Improved text cleaning with better sentence splitting"""
        # Remove markdown headers and empty lines
        text = "\n".join([
            line for line in text.split("\n") 
            if len(line.strip()) > 0 and not line.startswith(" = ")
        ])
        
        # Add space around punctuation
        text = re.sub(r'([.!?])', r' \1 ', text)
        
        # Convert to lowercase and clean special chars
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\.,;?!\'\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Add explicit sentence boundaries
        text = re.sub(r'([.!?])', r' \1 <eos> ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def build_vocab(self, text):
        """Create vocabulary from cleaned text"""
        words = text.split()
        word_counts = Counter(words)
        
        # Keep most common words including special tokens
        common_words = word_counts.most_common(self.vocab_size - 3)
        for idx, (word, _) in enumerate(common_words, start=3):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def text_to_sequences(self, text):
        """Convert text to numerical sequences with padding"""
        words = text.split()
        sequences = []
        current_seq = []
        
        for word in words:
            token = self.word2idx.get(word, self.word2idx["<unk>"])
            current_seq.append(token)
            
            if len(current_seq) == self.seq_length:
                sequences.append(current_seq)
                current_seq = []
        
        # Pad last sequence if needed
        if len(current_seq) > 0:
            pad_len = self.seq_length - len(current_seq)
            sequences.append(current_seq + [self.word2idx["<pad>"]] * pad_len)
            
        return sequences

    def process_data(self):
        """Full data processing pipeline"""
        # Get and clean text
        raw_text = self.get_wiki_text()
        cleaned_text = self.clean_text(raw_text)
        
        # Split into sentences using <eos> markers
        sentences = [s.strip() for s in cleaned_text.split("<eos>") if len(s.strip()) > 0]
        
        # Validate we have enough sentences
        if len(sentences) < 10:
            print("Warning: Very few sentences detected!")
            print("Sample sentences:", sentences[:3])
            raise ValueError("Insufficient sentences for training. Check data source and cleaning.")
        
        # Split into train/validation
        if len(sentences) > 100:
            train_sentences, val_sentences = train_test_split(
                sentences, 
                test_size=0.2,
                shuffle=True,
                random_state=42
            )
        else:
            # Fallback for small datasets
            split_idx = int(0.8 * len(sentences))
            train_sentences = sentences[:split_idx]
            val_sentences = sentences[split_idx:]
        
        # Build vocabulary from training data only
        self.build_vocab(" ".join(train_sentences))
        
        # Create sequences
        train_sequences = []
        for sent in train_sentences:
            train_sequences.extend(self.text_to_sequences(sent))
            
        val_sequences = []
        for sent in val_sentences:
            val_sequences.extend(self.text_to_sequences(sent))
        
        # Save processed data
        torch.save({
            "train": torch.LongTensor(train_sequences),
            "val": torch.LongTensor(val_sequences),
            "word2idx": self.word2idx,
            "idx2word": self.idx2word,
            "seq_length": self.seq_length
        }, "processed_data.pt")
        
        print("\nData preparation complete!")
        print(f"- Vocabulary size: {len(self.word2idx)}")
        print(f"- Training sequences: {len(train_sequences)}")
        print(f"- Validation sequences: {len(val_sequences)}")
        print(f"- Sequence length: {self.seq_length}")
        print("Saved to processed_data.pt")

if __name__ == "__main__":
    print("Starting data preparation...")
    processor = TextPreprocessor(
        vocab_size=20000,
        seq_length=50
    )
    processor.process_data()