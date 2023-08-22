import intel_extension_for_pytorch as ipex
import sys
import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F
print(f'Intel PyTorch Extension Version: {ipex.__version__}')


device = 'xpu' if torch.xpu.is_available() else ''
assert device == 'xpu', 'XPU not available'
batch_size = 32
block_size = 128
max_iterations = 10000
eval_interval = 500
eval_iterations = 200
learning_rate = 3e-4
n_embed = 384
n_head = 6
n_layer = 6
dropout = 0.2
output_size = 5000
torch.manual_seed(1337)


# Read dataset and tokenize
with open('foundation.txt', 'r', encoding='utf-8') as f:
	text = f.read()
enc = tiktoken.get_encoding('gpt2')
vocab_size = enc.n_vocab


# Split training and validation sets
data = torch.tensor(enc.encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


# Load data
def get_batch(split):
	data = train_data if split == 'train' else val_data
	ix = torch.randint(len(data) - block_size, (batch_size,))
	x = torch.stack([data[i:i+block_size] for i in ix])
	y = torch.stack([data[i+1:i+block_size+1] for i in ix])
	x, y = x.to(device), y.to(device)
	return x, y


# Estimate loss
@torch.no_grad()
def estimate_loss():
	out = {}
	model.eval()
	for split in ['train', 'val']:
		losses = torch.zeros(eval_iterations)
		for k in range(eval_iterations):
			x, y = get_batch(split)
			logits, loss = model(x, y)
			losses[k] = loss.item()
		out[split] = losses.mean()
	model.train()
	return out


class Head(nn.Module):
	def __init__(self, head_size):
		super().__init__()
		self.key = nn.Linear(n_embed, head_size, bias=False)
		self.query = nn.Linear(n_embed, head_size, bias=False)
		self.value = nn.Linear(n_embed, head_size, bias=False)
		self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		B, T, C = x.shape
		k = self.key(x)
		q = self.query(x)
		wei = q @ k.transpose(-2, -1) * C**-0.5
		wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
		wei = F.softmax(wei, dim = -1)
		wei = self.dropout(wei)
		v = self.value(x)
		out = wei @ v
		return out


class MultiHeadAttention(nn.Module):
	def __init__(self, number_heads, head_size):
		super().__init__()
		self.heads = nn.ModuleList([Head(head_size) for _ in range(number_heads)])
		self.projection = nn.Linear(n_embed, n_embed)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		out = torch.cat([h(x) for h in self.heads], dim = -1)
		out = self.dropout(self.projection(out))
		return out


class FeedForward(nn.Module):
	def __init__(self, n_embed):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(n_embed, 4 * n_embed),
			nn.ReLU(),
			nn.Linear(4 * n_embed, n_embed),
			nn.Dropout(dropout)
		)
	
	def forward(self, x):
		return self.net(x)


class Block(nn.Module):
	def __init__(self, n_embed, n_head):
		super().__init__()
		head_size = n_embed // n_head
		self.sa = MultiHeadAttention(n_head, head_size)
		self.ffwd = FeedForward(n_embed)
		self.ln1 = nn.LayerNorm(n_embed)
		self.ln2 = nn.LayerNorm(n_embed)

	def forward(self, x):
		x = x + self.sa(self.ln1(x))
		x = x + self.ffwd(self.ln2(x))
		return x


class BigramLanguageModel(nn.Module):
	def __init__(self):
		# Each token reads off logits for the next from lookup table
		super().__init__()
		self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
		self.position_embedding_table = nn.Embedding(block_size, n_embed)
		self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
		self.ln_f = nn.LayerNorm(n_embed) # Final layer norm
		self.lm_head = nn.Linear(n_embed, vocab_size)

	def forward(self, idx, targets=None):
		B, T = idx.shape
		# idx and targets are both (B,T) tensor of integers
		token_embeddings = self.token_embedding_table(idx) # (Batch, Time, Channels)
		position_embeddings = self.position_embedding_table(torch.arange(T, device=device)) # (Time, Channels)
		x = token_embeddings + position_embeddings
		x = self.blocks(x)
		x = self.ln_f(x)
		logits = self.lm_head(x) # (Batch, Time, vocab_size)

		if targets is None:
			loss = None
		else:
			B, T, C = logits.shape
			logits = logits.view(B*T, C)
			targets = targets.view(B*T)
			loss = F.cross_entropy(logits, targets)

		return logits, loss

	def generate(self, idx, max_new_tokens):
		# idx is (B, T) array of indices in the current context
		for _ in range(max_new_tokens):
			# Crop idx to the last block_size tokens
			idx_cond = idx[:, -block_size:]
			# Get the predictions
			logits, loss = self(idx_cond)
			# Focus on the last time step
			logits = logits[:, -1, :] # becomes (B, C)
			# Apply softmax to get probabilities
			probs = F.softmax(logits, dim=-1) # becomes (B, C)
			# Sample from the distribution
			idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
			# Append sampled index to the running sequence
			idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
		return idx


# Check if loading pretrained model
if len(sys.argv) > 1:
	print("Loading pretrained model")
	model = BigramLanguageModel()
	model = model.to(device)
	model.load_state_dict(torch.load(str(sys.argv[1])))
	model.eval()

else:
	model = BigramLanguageModel()
	model = model.to(device)
	optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
	total_memory = torch.xpu.get_device_properties(0).total_memory // 1024 ** 2

	for i in range(max_iterations):
		if i % eval_interval == 0 or i == max_iterations-1:
			losses = estimate_loss()
			allocated = torch.xpu.memory_allocated(0) // 1024 ** 2
			reserved = torch.xpu.memory_reserved(0) // 1024 ** 2
			print(f"Step {i}: Training loss {losses['train']:.4f}, Validation loss {losses['val']:.4f}")
			print(f"Memory: {allocated}MB / {reserved}MB / {total_memory}MB")

		# Sample a batch of data and evaluate loss
		xb, yb = get_batch('train')
		logits, loss = model(xb, yb)
		optimizer.zero_grad(set_to_none=True)
		loss.backward()
		optimizer.step()
	
	torch.save(model.state_dict(), 'model.pt')


# Generate output from model and save to file
context = torch.zeros((1, 1), dtype=torch.long, device=device)
output = enc.decode(model.generate(context, max_new_tokens=output_size)[0].tolist())
f = open('output.txt', 'w')
f.write(output)
f.close()
