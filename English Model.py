import re
import requests
from bs4 import BeautifulSoup
import torch
import torch.nn as nn
import torch.optim as optim
import random
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from difflib import SequenceMatcher
from sklearn.metrics import jaccard_score
import Levenshtein

# 1. GPU or CPU?
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_pokemon_names(filename="pokemon_en_clean.txt"):
    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"File '{filename}' not found. Please download it from the provided link and place it in the same folder as this script."
        )
    with open(filename, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(names)} Pokémon names from '{filename}'.")
    return names

# Call this in your script instead of web scraping
english_names = load_pokemon_names()

# ─────────────────────────────────────────────────────────────
# 3. LOAD REAL NAMES
# ─────────────────────────────────────────────────────────────
def load_real_names(filename="pokemon_en_clean.txt"):
    with open(filename, "r", encoding="utf-8") as file:
        return [line.strip() for line in file]

english_names = load_real_names()

# Create vocab + mappings
english_chars = sorted(set("".join(english_names)))
english_stoi = {char: i for i, char in enumerate(english_chars)}
english_itos = {i: char for char, i in english_stoi.items()}

# Hyperparams
input_size_en = len(english_chars)
hidden_size = 512
learning_rate = 0.0003
num_epochs = 10000

# ─────────────────────────────────────────────────────────────
# 4. LSTM MODEL DEFINITION
# ─────────────────────────────────────────────────────────────
class LSTMPokemonNameGenerator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMPokemonNameGenerator, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=3, batch_first=True, dropout=0.25)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out[:, -1, :])  # last time step
        return self.softmax(out), hidden

    def init_hidden(self):
        return (torch.zeros(3, 1, self.hidden_size).to(device),
                torch.zeros(3, 1, self.hidden_size).to(device))

# ─────────────────────────────────────────────────────────────
# 5. HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────
def char_to_tensor(char, stoi):
    tensor = torch.zeros(1, 1, len(stoi), device=device)
    tensor[0][0][stoi[char]] = 1
    return tensor

def name_to_tensor(name, stoi):
    return torch.cat([char_to_tensor(c, stoi) for c in name], dim=0)

def smoothed_nll_loss(predictions, target, smoothing=0.1):
    """Label smoothing."""
    confidence = 1.0 - smoothing
    n_classes = predictions.size(-1)
    one_hot = torch.zeros_like(predictions).scatter(1, target.unsqueeze(1), 1)
    smooth_target = one_hot * confidence + (1 - one_hot) * (smoothing / (n_classes - 1))
    return torch.mean(-smooth_target * predictions)

# ─────────────────────────────────────────────────────────────
# 6. TRAINING LOOP
# ─────────────────────────────────────────────────────────────
def train(model, names, stoi, num_epochs=num_epochs):
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3000, T_mult=2, eta_min=1e-6)
    model.train()

    for epoch in range(num_epochs):
        name = random.choice(names)
        hidden = model.init_hidden()
        optimizer.zero_grad()
        loss = 0.0

        input_tensor = name_to_tensor(name[:-1], stoi)
        target_tensor = torch.tensor([stoi[c] for c in name[1:]], device=device)

        for i in range(len(name) - 1):
            output, hidden = model(input_tensor[i].unsqueeze(0), hidden)
            loss += smoothed_nll_loss(output, target_tensor[i].unsqueeze(0))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    print("Training Complete!")

# ─────────────────────────────────────────────────────────────
# 7. GENERATE NAMES
# ─────────────────────────────────────────────────────────────
def generate_name(model, stoi, itos, start_char, min_length=4, max_length=12):
    model.eval()
    with torch.no_grad():
        hidden = model.init_hidden()
        input_tensor = char_to_tensor(start_char.upper(), stoi)
        name = start_char.upper()

        length_options = [5, 7, 9, 12]
        chosen_length = random.choice(length_options)
        temperature = random.choice([0.7, 0.8, 0.9])

        for i in range(chosen_length - 1):
            output, hidden = model(input_tensor, hidden)
            output_dist = output.div(temperature).exp()
            topi = torch.multinomial(output_dist, 1)[0]
            next_char = itos[topi.item()].lower()

            stop_probability = min(0.08 + (i / chosen_length) * 0.2, 0.27)
            if i >= min_length and random.random() < stop_probability:
                break

            name += next_char
            input_tensor = char_to_tensor(next_char, stoi)

        name = name.capitalize()

        # suffixes
        suffixes = ["chu", "mon", "tar", "zard", "to", "ine", "ite", "saur", "gon", "dran", "lux"]
        if random.random() < 0.2:
            name += random.choice(suffixes)

    return name

def generate_names(model, stoi, itos, num_names=25):
    return [
        generate_name(model, stoi, itos, random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
        for _ in range(num_names)
    ]

# ─────────────────────────────────────────────────────────────
# 8. EVALUATION METRICS
# ─────────────────────────────────────────────────────────────
def avg_levenshtein_distance(real_names, generated_names):
    distances = [
        min(Levenshtein.distance(gen.lower(), real.lower()) for real in real_names)
        for gen in generated_names
    ]
    return np.mean(distances)

def avg_jaccard_similarity(real_names, generated_names):
    """Computes average Jaccard similarity using character sets."""
    def char_set(name):
        return set(name.lower())  # Ensure lowercase to avoid mismatches

    similarities = []
    for gen in generated_names:
        gen_chars = list(char_set(gen))  # Convert set to sorted list

        max_sim = 0  # Store the highest Jaccard similarity for this generated name
        for real in real_names:
            real_chars = list(char_set(real))  # Convert set to sorted list

            # Ensure both vectors have the same length by padding
            max_len = max(len(gen_chars), len(real_chars))
            gen_chars_padded = gen_chars + [''] * (max_len - len(gen_chars))
            real_chars_padded = real_chars + [''] * (max_len - len(real_chars))

            # Compute Jaccard similarity
            sim = jaccard_score(gen_chars_padded, real_chars_padded, average='macro')
            max_sim = max(max_sim, sim)  # Keep the highest similarity

        similarities.append(max_sim)

    return np.mean(similarities)


def length_distribution(real_names, generated_names):
    real_lengths = [len(n) for n in real_names]
    gen_lengths = [len(n) for n in generated_names]

    plt.hist(real_lengths, alpha=0.6, label="Real Names", bins=10)
    plt.hist(gen_lengths, alpha=0.6, label="Generated Names", bins=10)
    plt.xlabel("Name Length")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title("Pokémon Name Length Distribution")
    plt.show()

def is_vowel_char(c):
    return c in "aeiou"

def phonetic_balance_check(names):
    """Checks consonant-vowel alternation in each name."""
    scores = []
    for nm in names:
        nm = nm.lower()
        if len(nm) <= 1:
            scores.append(1.0)
            continue
        transitions = 0
        for i in range(len(nm) - 1):
            if is_vowel_char(nm[i]) != is_vowel_char(nm[i+1]):
                transitions += 1
        scores.append(transitions / (len(nm) - 1))
    return np.mean(scores)

def evaluate_model(model, stoi, itos, num_samples=50):
    real_names = load_real_names()
    generated_names = generate_names(model, stoi, itos, num_names=num_samples)

    lev_dist = avg_levenshtein_distance(real_names, generated_names)
    jaccard_sim = avg_jaccard_similarity(real_names, generated_names)
    phone_balance = phonetic_balance_check(generated_names)

    # Print results
    print(f"\nEvaluation Results:")
    print(f"- Average Levenshtein Distance: {lev_dist:.4f} (lower = more novel)")
    print(f"- Average Jaccard Similarity:  {jaccard_sim:.4f} (higher = more similar chars)")
    print(f"- Phonetic Balance Score:      {phone_balance:.4f} (0-1 range, higher = smoother CV alternation)")

    # Optionally plot length distribution
    length_distribution(real_names, generated_names)

    return generated_names

# ─────────────────────────────────────────────────────────────
# 9. RUN TRAINING & EVALUATION
# ─────────────────────────────────────────────────────────────
# 1. Initialize model
english_model = LSTMPokemonNameGenerator(input_size_en, hidden_size, input_size_en).to(device)

# 2. (Optional) compile if PyTorch 2.0+ is available
try:
    english_model = torch.compile(english_model)
except:
    print("torch.compile() not available, continuing without it.")

# 3. Train the model
train(english_model, english_names, english_stoi, num_epochs=num_epochs)

# 4. Generate sample names
print("\nGenerated English Pokémon Names:")
for _ in range(10):
    print(generate_name(english_model, english_stoi, english_itos, random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")))

# 5. Evaluate
_ = evaluate_model(english_model, english_stoi, english_itos, num_samples=50)
