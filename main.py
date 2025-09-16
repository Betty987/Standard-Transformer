import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import os
import pickle
import json
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from bert_score import score as bertscore
import time
import tiktoken
from contextlib import nullcontext
nltk.download('punkt')
import os
import requests

import numpy as np

# Hyperparameters

batch_size = 8
# vocab_size = 50340
block_size = 256
MAX_LENGTH = 256
learning_rate = 1e-3
n_embd = 64
n_head = 8
n_layer = 6
dropout = 0.1
max_epochs = 20
max_new_tokens = 200
temperature = 0.8


# Directory setup

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
DATA_DIR = os.path.join(BASE_DIR, "ptbdataset") 
TOKENIZER_DIR = os.path.join(os.path.dirname(DATA_DIR), 'tokenized_ptb')
os.makedirs(TOKENIZER_DIR, exist_ok=True)

# # BPE Tokenizer class
# class BPETokenizer:
#     def __init__(self):
#         self.tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
#         self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
#         self.special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[EOS]"]

#     def train_and_save(self):
#         tokenizer_path = os.path.join(TOKENIZER_DIR, "tokenizer.json")
#         if os.path.exists(tokenizer_path):
#             print(f"Tokenizer already exists at {tokenizer_path}, skipping training.")
#             return
        
#         train_path = os.path.join(DATA_DIR, "ptb.train.txt")
#         if not os.path.exists(train_path):
#             raise FileNotFoundError(f"Train file not found: {train_path}")
        
#         with open(train_path, "r", encoding="utf-8") as f:
#             sentences = [line.strip() for line in f if line.strip()]

#         trainer = trainers.BpeTrainer(
#             special_tokens=self.special_tokens,
#             vocab_size=4000,
#             min_frequency=2
#         )
#         self.tokenizer.train_from_iterator(sentences, trainer=trainer)
#         tokenizer_path = os.path.join(TOKENIZER_DIR, "tokenizer.json")
#         self.tokenizer.save(tokenizer_path)

#         metadata = {"special_tokens": self.special_tokens}
#         metadata_path = os.path.join(TOKENIZER_DIR, "metadata.json")
#         with open(metadata_path, "w", encoding="utf-8") as f:
#             json.dump(metadata, f, indent=4)

#         print(f"Tokenizer trained and saved to {tokenizer_path}")
#         print(f"Metadata saved to {metadata_path}")

#     def tokenize_and_save(self, subset_name):
#         tokenizer_path = os.path.join(TOKENIZER_DIR, "tokenizer.json")
#         if not os.path.exists(tokenizer_path):
#             raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
#         self.tokenizer = Tokenizer.from_file(tokenizer_path)
        
#         subset_path = os.path.join(DATA_DIR, f"ptb.{subset_name}.txt")
#         if not os.path.exists(subset_path):
#             raise FileNotFoundError(f"{subset_name}.txt not found in {DATA_DIR}")
        
#         with open(subset_path, "r", encoding="utf-8") as f:
#             sep_id = self.tokenizer.token_to_id("[EOS]")
#             if sep_id is None:
#                 raise ValueError("Special token [EOS] not found in tokenizer vocabulary.")
            
#             tokenized = [
#                 self.tokenizer.encode(line.strip()).ids + [sep_id]
#                 for line in f if line.strip()
#             ]

#         output_path = os.path.join(TOKENIZER_DIR, f"{subset_name}_ids.pkl")
#         if os.path.exists(output_path):
#             print(f"Tokenized IDs already exist for {subset_name} at {output_path}, skipping.")
#             return 

#         with open(output_path, "wb") as f:
#             pickle.dump(tokenized, f)

#         print(f"Tokenized ptb.{subset_name}.txt and saved IDs to {output_path}")
# # Initialize and train tokenizer
# bpe = BPETokenizer()
# bpe.train_and_save()

# # Tokenize datasets
# bpe.tokenize_and_save("train")
# bpe.tokenize_and_save("valid")
# bpe.tokenize_and_save("test")        

# # Penn Treebank Dataset class
# class PennTreebankDataset(Dataset):
#     def __init__(self, tokenized_file, tokenizer_dir, block_size):
#         self.tokenizer_dir = tokenizer_dir
#         self.block_size = block_size

#         tokenized_file_path = os.path.join(self.tokenizer_dir, tokenized_file)
#         if not os.path.exists(tokenized_file_path):
#             raise FileNotFoundError(f"Tokenized file not found: {tokenized_file_path}")

#         with open(tokenized_file_path, 'rb') as f:
#             self.sequences = pickle.load(f)

#         self.sequences = [seq for seq in self.sequences if len(seq) > 1]

#     def __len__(self):
#         return len(self.sequences)

#     def __getitem__(self, idx):
#         seq = self.sequences[idx]
#         input_ids = torch.tensor(seq[:-1][:self.block_size], dtype=torch.long)
#         target_ids = torch.tensor(seq[1:][:self.block_size], dtype=torch.long)
        
#         return {"input_ids": input_ids, "target_ids": target_ids}

# # Create datasets with Subset
# train_dataset = PennTreebankDataset("train_ids.pkl", TOKENIZER_DIR, MAX_LENGTH)
# # train_dataset = Subset(train_dataset, range(min(len(train_dataset), 50000)))
# val_dataset = PennTreebankDataset("valid_ids.pkl", TOKENIZER_DIR,  MAX_LENGTH)
# test_dataset = PennTreebankDataset("test_ids.pkl", TOKENIZER_DIR,  MAX_LENGTH)
# # test_dataset = Subset(test_dataset, range(min(len(test_dataset), 25000)))

# # Create data loaders
# train_loader = DataLoader(
#     train_dataset,
#     batch_size=batch_size,
#     shuffle=True,
#     collate_fn=lambda batch: pad_collate_fn(batch, pad_token_id)
# )
# valid_loader = DataLoader(
#     val_dataset,
#     batch_size=batch_size,
#     collate_fn=lambda batch: pad_collate_fn(batch, pad_token_id)
# )
# test_loader = DataLoader(
#     test_dataset,
#     batch_size=batch_size,
#     collate_fn=lambda batch: pad_collate_fn(batch, pad_token_id)
# )
# # Padding collate function
# def pad_collate_fn(batch, pad_token_id=0):
#     input_seqs = [item["input_ids"] for item in batch]
#     target_seqs = [item["target_ids"] for item in batch]

#     input_seqs = pad_sequence(input_seqs, batch_first=True, padding_value=pad_token_id)
#     target_seqs = pad_sequence(target_seqs, batch_first=True, padding_value=pad_token_id)

#     return {"input_ids": input_seqs, "target_ids": target_seqs}

# def decode_ids(tokenizer, ids, stop_at_eos = True):
#     text = tokenizer.decode(ids, skip_special_tokens=True)
#     if stop_at_eos and "[EOS]" in text:
#         text = text.split("[EOS]")[0].strip()
#     return text

# # Load tokenizer function
# def load_tokenizer():
#     tokenizer_path = os.path.join(TOKENIZER_DIR, "tokenizer.json")
#     return Tokenizer.from_file(tokenizer_path)


# # Load tokenizer
# tokenizer = load_tokenizer()
# vocab_size = tokenizer.get_vocab_size()
# pad_token_id = tokenizer.token_to_id("[PAD]")
# eos_token_id = tokenizer.token_to_id("[EOS]")
# download the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w', encoding='utf-8') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files in 'data' directory
data_dir = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(data_dir, exist_ok=True)
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(data_dir, 'train.bin'))
val_ids.tofile(os.path.join(data_dir, 'val.bin'))
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# Model architecture=casual self-attention
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):#mlp
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.ln1 = nn.LayerNorm(n_embd)
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T))
        x = self.dropout(tok_emb + pos_emb)
        x = self.blocks(x)
        x = self.ln_f(x)
        # if targets is None:
        #     loss = None
        # else:
        #     B, T, C = logits.shape
        #     logits_flat = logits.view(B*T, C)
        #     targets_flat = targets.view(B*T)
        #     loss = F.cross_entropy(logits_flat, targets_flat)
        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


# Training function
def train(model, optimizer, epoch, num_batches=100):
    model.train()
    total_loss = 0
    total_batches = 0
    for batch_idx in range(num_batches):
        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_batches += 1
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{num_batches} | train_loss {loss.item():.4f} | train_perplexity {torch.exp(loss).item():.4f}", flush=True)
    avg_loss = total_loss / total_batches
    avg_perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return avg_loss, avg_perplexity


def compute_text_metrics(predictions, targets):
    print("\nComputing BERTScore and BLEU...")
    P, R, F1 = bertscore(
        predictions,
        targets,
        lang="en",
        model_type="roberta-base",
        rescale_with_baseline=True,
    )
    print(f"BERTScore (F1): {F1.mean().item():.4f}")

    smooth_fn = SmoothingFunction().method4
    tokenized_targets = [[target.split()] for target in targets]
    tokenized_pred = [pred.split() for pred in predictions]
    bleu = corpus_bleu(tokenized_targets, tokenized_pred, smoothing_function=smooth_fn)
    print(f"BLEU Score: {bleu:.4f}")

@torch.no_grad()
def evaluate(model, test_loader, tokenizer, max_batches=None, compute_metrics=True):
    start_time = time.time()
    model.eval()
    total_loss = 0
    total_batches = 0
    pad_token_id = tokenizer.token_to_id("[PAD]")
    decoded_targets, decoded_predictions = [], []
    if max_batches is None:
        print(f"Evaluating on the full test set...")
    else:
        print(f"Evaluating on up to {max_batches} batches...")

    for batch_idx, batch in enumerate(test_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        input_ids = batch['input_ids']
        targets = batch['target_ids']

        # Compute loss
        logits, loss = model(input_ids, targets)
        total_loss += loss.item()
        total_batches += 1

        # if compute_metrics:
        #      preds = torch.argmax(logits, dim=-1)
        #      mask = targets != pad_token_id
        #      for i in range(preds.size(0)):
        #         pred_str = decode_ids(tokenizer, preds[i][mask[i]].tolist(), stop_at_eos=True)
        #         tgt_str = decode_ids(tokenizer, targets[i][mask[i]].tolist(), stop_at_eos=True)
        #         decoded_predictions.append(pred_str)
        #         decoded_targets.append(tgt_str)

           
    # if compute_metrics and decoded_predictions and decoded_targets:
    #     compute_text_metrics(decoded_predictions, decoded_targets)
           

    # Compute average loss and perplexity
    avg_loss = total_loss / total_batches if total_batches > 0 else float('inf')
    avg_perplexity = torch.exp(torch.tensor(avg_loss)).item() if avg_loss != float('inf') else float('inf')
    elapsed = time.time() - start_time
    print(f"Evaluation completed in {elapsed:.2f} seconds")
    print(f"Total Batches Processed: {batch_idx + 1}")
    print(f"Avg Test CE Loss: {avg_loss:.4f} | Avg Test Perplexity: {avg_perplexity:.4f}")
    return avg_loss,avg_perplexity



# Training phase
vocab_size = enc.n_vocab 
model = LanguageModel()
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("Starting training...")
start_training_time = time.time()
for epoch in range(max_epochs):
    print(f"\nEpoch {epoch + 1}/{max_epochs}")
    avg_loss, avg_perplexity = train(model, optimizer, epoch, num_batches=100)
    print(f"Epoch {epoch + 1} completed | avg_train_loss {avg_loss:.4f} | avg_train_perplexity {avg_perplexity:.4f}")
total_training_time = time.time() - start_training_time
print(f"Total Training Time: {total_training_time:.2f} seconds", flush=True)
print("========== Training completed ==========", flush=True)
# Save model
save_path = "checkpoints/gpt_backprop.pt"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
if os.path.exists(save_path):
    os.remove(save_path)
torch.save({"model_state": model.state_dict()}, save_path)
print("Model saved.")



# Evaluate with metrics
# test_loss, test_perplexity = evaluate(model, max_batches=10)

@torch.no_grad()
def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
    """
    Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    """
    for _ in range(max_new_tokens):
        # if the sequence context is growing too long we must crop it at block_size
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        # forward the model to get the logits for the index in the sequence
        logits, _ = self(idx_cond)
        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        # apply softmax to convert logits to (normalized) probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1)
        # append sampled index to the running sequence and continue
        idx = torch.cat((idx, idx_next), dim=1)

    return idx
num_samples = 10
# Generation setup
top_k = 200
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = torch.float32
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)
# encode the beginning of the prompt
start = "\n"
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print('---------------')






# # Generate samples
def generate(self, input_ids, max_new_tokens=50, temperature=1.0):
            model.eval()
            input_tensor = input_ids.unsqueeze(0) # Add batch dimension

            for _ in range(max_new_tokens):
                if input_tensor.size(1) > block_size:
                    input_tensor = input_tensor[:, -block_size:]
                with torch.no_grad():
                    logits, _ = self(input_tensor)
                    logits = logits[:, -1, :] / temperature
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    input_tensor = torch.cat((input_tensor, next_token), dim=1)
                # if next_token.item() == eos_token_id:
                #     break
            
            return input_tensor[0] 
for batch_idx, batch in enumerate(test_loader):
    input_ids = batch["input_ids"]
    target_ids = batch["target_ids"]
    break 

num_samples = 5
prompt_len = 4
i = 64

for i in range(num_samples):
    prompt_ids = input_ids[i][:prompt_len]
    generated_ids = generate(model,prompt_ids, max_new_tokens= 50, temperature=0.7)

    target_continuation = target_ids[i][prompt_len:]
    target_continuation = target_continuation[target_continuation != pad_token_id].tolist()

    generated_continuation = generated_ids[prompt_len:].tolist()

    # Decode all
    prompt_str = decode_ids(tokenizer, prompt_ids.tolist())
    target_str = decode_ids(tokenizer, target_continuation, stop_at_eos=True)
    predict_str = decode_ids(tokenizer, generated_continuation, stop_at_eos=True)

    print(f"\n[Batch {batch_idx + 1}, Sample {i + 1}]")
    print(f"[PROMPT ]: {prompt_str}")
    print(f"[TARGET ]: {target_str}")
    print(f"[PREDICT]: {predict_str}")
