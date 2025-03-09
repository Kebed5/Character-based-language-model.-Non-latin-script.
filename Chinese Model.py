import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_pokemon_names(filename="pokemon_cn_clean.txt"):
    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"File '{filename}' not found. Please download it and place it in the script's directory."
        )
    with open(filename, encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

chinese_names = load_pokemon_names()


augmented_names = []
for name in chinese_names:
    
    augmented_names.append(name)

    if len(name) >= 3:
        augmented_names.append(name[1:] + name[0])

    if len(name) >= 4:
        augmented_names.append(name[:-1])
names = augmented_names

special_tokens = ['<s>', '</s>', '<pad>']
unique_chars = set(ch for name in names for ch in name)
chars = special_tokens + sorted(unique_chars)

char2idx = {ch: i for i, ch in enumerate(chars)}
idx2char = {i: ch for ch, i in char2idx.items()}
vocab_size = len(chars)

class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_size=256, n_layers=3):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers=n_layers, dropout=0.6, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden, lengths):
        emb = self.embedding(x)
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, hidden = self.lstm(packed, hidden)
        output, _ = pad_packed_sequence(packed_out, batch_first=True)
        logits = self.fc(output)
        return logits, hidden

    def init_hidden(self, batch_size):
        return (
            torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device),
            torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device)
        )

def batch_generator(names, batch_size=128):
    sequences = [torch.tensor([char2idx['<s>']] + [char2idx[ch] for ch in name] + [char2idx['</s>']]) for name in names]
    random.shuffle(sequences)
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i+batch_size]
        lengths = torch.tensor([len(seq)-1 for seq in batch])
        inputs = pad_sequence([seq[:-1] for seq in batch], batch_first=True, padding_value=char2idx['<pad>']).to(device)
        targets = pad_sequence([seq[1:] for seq in batch], batch_first=True, padding_value=char2idx['<pad>']).to(device)
        lengths, perm_idx = lengths.sort(0, descending=True)
        yield inputs[perm_idx], targets[perm_idx], lengths

def train(model, names, epochs=150):
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=char2idx['<pad>'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for inputs, targets, lengths in batch_generator(names):
            optimizer.zero_grad()
            hidden = model.init_hidden(inputs.size(0))
            output, hidden = model(inputs, hidden, lengths)
            loss = criterion(output.view(-1, vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        if (epoch+1) % 50 == 0:
            print(f'Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f}')

def generate_name_top_p(model, char2idx, idx2char, device, max_length=8, temperature=1.0, top_p=0.8):
    model.eval()
    with torch.no_grad():
        input_idx = torch.tensor([[char2idx['<s>']]], device=device)
        hidden = model.init_hidden(1)
        generated_name = []

        for _ in range(max_length):
            output, hidden = model(input_idx, hidden, torch.tensor([1]))
            probabilities = torch.softmax(output[:, -1, :] / temperature, dim=-1).squeeze()

            sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=0)

            cutoff = cumulative_probs > top_p
            if cutoff.any():
                cutoff_idx = cutoff.nonzero(as_tuple=True)[0][0] + 1
                sorted_probs = sorted_probs[:cutoff_idx]
                sorted_indices = sorted_indices[:cutoff_idx]

            next_idx = sorted_indices[torch.multinomial(sorted_probs, 1)].item()
            next_char = idx2char[next_idx]

            if next_char == '</s>':
                break

            generated_name.append(next_char)
            input_idx = torch.tensor([[next_idx]], device=device)

        return ''.join(generated_name)

def compute_perplexity(model, names):
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=char2idx['<pad>'])
    total_loss, total_chars = 0, 0
    with torch.no_grad():
        for inputs, targets, lengths in batch_generator(names):
            hidden = model.init_hidden(inputs.size(0))
            output, hidden = model(inputs, hidden, lengths)
            loss = criterion(output.view(-1, vocab_size), targets.view(-1))
            total_loss += loss.item() * lengths.sum().item()
            total_chars += lengths.sum().item()
    perplexity = torch.exp(torch.tensor(total_loss / total_chars))
    return perplexity.item()

def compute_accuracy(model, names):
    model.eval()
    correct, total = 0, 0
    criterion = nn.CrossEntropyLoss(ignore_index=char2idx['<pad>'], reduction='sum')
    
    for inputs, targets, lengths in batch_generator(names):
        hidden = model.init_hidden(inputs.size(0))
        outputs, hidden = model(inputs, hidden, lengths)
        predictions = outputs.argmax(dim=-1)
        mask = (targets != char2idx['<pad>'])
        correct += (predictions == targets).masked_select(mask).sum().item()
        total += mask.sum().item()

    return correct / total

if __name__ == "__main__":
    random.shuffle(names)
    split_idx = int(0.9 * len(names))
    train_names = names[:split_idx]
    test_names = names[split_idx:]

    model = CharLSTM(vocab_size, embed_dim=64, hidden_size=128, n_layers=2).to(device)
    train(model, train_names, epochs=300)

    print(f'\nTest Perplexity: {compute_perplexity(model, test_names):.4f}')
    print(f'Test Accuracy: {compute_accuracy(model, test_names):.2%}')

    print("\nGenerated Pokémon Names (top-p sampling):")
    for _ in range(50):
        print(generate_name_top_p(model, char2idx, idx2char, device, temperature=1.0, top_p=0.8))
        