# NLP Pretraining Data Pipeline (BPE + PyTorch)

This project implements a **minimal NLP data pipeline inspired by language model pre-training workflows**.  
It demonstrates how raw text is transformed into numerical batches ready for neural network training.

The pipeline includes:
- Text preprocessing
- Training a simple Byte Pair Encoding (BPE) tokenizer
- Tokenization and numerical encoding
- Sliding window dataset creation
- Batching with PyTorch DataLoader

The goal of this repository is **educational**: to illustrate the full data preparation pipeline used before training models such as Transformers.

---

# Pipeline Overview

Raw Text → Tokenize (Simple BPE) → Numericalize → Padding → Batching → DataLoader

---

# Final Pipeline Architecture

Raw Text  
↓  
Preprocess  
↓  
Train BPE  
↓  
Tokenizer.encode()  
↓  
Numerical IDs  
↓  
TextDataset (Sliding Window)  
↓  
DataLoader (Batching + Shuffle)  
↓  
Model Training  

---

# Components

## 1. Text Preprocessing
The raw text is normalized and split into words.

## 2. BPE Training
A simple **Byte Pair Encoding (BPE)** algorithm is trained on the corpus to learn subword merges.

## 3. Tokenization
Words are decomposed into subword tokens using the learned merges.

## 4. Numerical Encoding
Tokens are converted into **integer IDs** using a token vocabulary.

## 5. Dataset Construction
A `TextDataset` class generates training samples using a **sliding window approach**.

Example:

Input sequence  
[t1 t2 t3 t4]

Target sequence  
[t2 t3 t4 t5]

This is the typical format used for **next-token prediction in language models**.

## 6. DataLoader
The dataset is wrapped with a **PyTorch DataLoader** that handles batching and shuffling during training.

---

# Example Output

Encoded tokens:
[5, 1, 7, 9, 4, 35, 21, 16, 4, 5, 1, 4, 9, 7]

Batch shapes:
Input: torch.Size([4, 8])
Target: torch.Size([4, 8])

Where:

- `4` = batch size
- `8` = sequence length (block size)

---

# Technologies

- Python
- PyTorch
- Custom Byte Pair Encoding (BPE)

---

# Educational Purpose

This repository focuses on **clarity and understanding**, not optimization.  
It demonstrates how modern NLP systems prepare text data before training neural networks.

The implementation is intentionally simple and modular so each step of the pipeline can be easily understood.
