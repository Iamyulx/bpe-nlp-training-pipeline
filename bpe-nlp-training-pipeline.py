raw_text = """ low lower newest widest low lowest newer """

def preprocess(text):
  return text.lower().strip().split()

def build_vocab(words):
  vocab = {}
  for word in words:
    tokens = list(word) + ['</w>']
    vocab[tuple(tokens)] = vocab.get(tuple(tokens), 0) + 1
  return vocab

def get_pair_frequencies(vocab):
  pairs = {}
  for word, freq in vocab.items():
    for i in range(len(word)-1):
      pair = (word[i], word[i+1])
      pairs[pair] = pairs.get(pair, 0) + freq
  return pairs

def merge_pair(pair, vocab):
  new_vocab = {}
  bigram = pair
  replacement = ''.join(pair)

  for word, freq in vocab.items():
    new_word = []
    i = 0
    while i < len(word):
      if i < len(word)-1 and (word[i], word[i+1]) == bigram:
        new_word.append(replacement)
        i += 2
      else:
        new_word.append(word[i])
        i += 1
    new_vocab[tuple(new_word)] = freq
  return new_vocab

def train_bpe(words, num_merges=10):
  vocab = build_vocab(words)
  merges = []

  for _ in range(num_merges):
    pairs = get_pair_frequencies(vocab)
    if not pairs:
      break
    best_pair = max(pairs, key=pairs.get)
    merges.append(best_pair)
    vocab = merge_pair(best_pair, vocab)
  return merges



class BPETokenizer:
    def __init__(self, merges):
        self.merges = merges
        self.vocab = {}
        self.build_token_vocab()

    def build_token_vocab(self):
        idx = 0
        # Add merged pairs as tokens
        for pair in self.merges:
            token = ''.join(pair)
            if token not in self.vocab:
                self.vocab[token] = idx
                idx += 1

        # Add individual characters
        # Collect all unique characters first
        all_chars = set()
        for pair in self.merges:
            for char in pair:
                if len(char) == 1: # Only add single characters
                    all_chars.add(char)
        for c in "abcdefghijklmnopqrstuvwxyz</w>": # Ensure all possible characters are covered
            all_chars.add(c)

        for c in sorted(list(all_chars)): # Sort for consistent vocabulary order
            if c not in self.vocab:
                self.vocab[c] = idx
                idx += 1



    def tokenize(self, word):
        tokens = list(word) + ['</w>']
        # Apply merges iteratively
        for pair in self.merges:
            i = 0
            while i < len(tokens) - 1:
                if (tokens[i], tokens[i+1]) == pair:
                    tokens[i:i+2] = [''.join(pair)]
                    # No i += 1 here, because the list has shrunk and the next potential pair starts at current i
                else:
                    i += 1
        return tokens

    def encode(self, text):
        words = preprocess(text)
        tokens = []
        for word in words:
            tokens.extend(self.tokenize(word))
        return [self.vocab.get(token, self.vocab['<unk>']) if token not in self.vocab else self.vocab[token] for token in tokens] # Added handling for unknown tokens


words = preprocess(raw_text)
merges = train_bpe(words, num_merges=10)
tokenizer = BPETokenizer(merges)

encoded = tokenizer.encode(raw_text)
print(encoded)


import torch
from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):
  def __init__(self, encoded_text, block_size):
    self.data = encoded_text
    self.block_size = block_size

  def __len__(self):
    return len(self.data) - self.block_size

  def __getitem__(self, idx):
    x = torch.tensor(self.data[idx:idx+self.block_size])
    y = torch.tensor(self.data[idx+1:idx+self.block_size+1])
    return x, y


# Input: [t1 t2 t3 t4]
# Target: [t2 t3 t4 t5]

block_size = 8
batch_size = 4

dataset = TextDataset(encoded, block_size)
dataloader = DataLoader(dataset, batch_size, shuffle=True)

for batch_x, batch_y in dataloader:
  print("Input:", batch_x.shape)
  print("Target:", batch_y.shape)
  break
