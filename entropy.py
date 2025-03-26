import numpy as np
import math
from collections import Counter

def shannon_entropy(data):
    """Calculate Shannon entropy of a dataset"""
    counter = Counter(data)
    total_count = len(data)
    probabilities = [count/total_count for count in counter.values()]
    return -sum(p * math.log2(p) for p in probabilities)

def min_entropy(data):
    """Calculate min-entropy of a dataset"""
    counter = Counter(data)
    total_count = len(data)
    max_probability = max(count/total_count for count in counter.values())
    return -math.log2(max_probability)

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
num_sets = 40
set_size = 1000
min_value = 0
max_value = 100

results = []

# Generate sets with different entropy levels
for i in range(num_sets):
    # Calculate mixing factor (0 means low entropy, 1 means high entropy)
    mixing_factor = i / (num_sets - 1)
    
    if mixing_factor < 0.5:
        # For lower entropy: concentrate values around a single number
        concentration = 1 - 2 * mixing_factor  # 1→0 as mixing_factor goes from 0→0.5
        primary_value = np.random.randint(min_value, max_value)
        
        # Determine how many values will be the primary value
        primary_count = int(concentration * set_size)
        
        # Generate the set
        data_set = np.full(set_size, primary_value)
        random_indices = np.random.choice(set_size, set_size - primary_count, replace=False)
        data_set[random_indices] = np.random.randint(min_value, max_value, size=set_size - primary_count)
    else:
        # For higher entropy: gradually approach uniform distribution
        uniformity = (mixing_factor - 0.5) * 2  # 0→1 as mixing_factor goes from 0.5→1
        
        # Number of unique values to use
        num_unique = min_value + int(uniformity * (max_value - min_value))
        if num_unique < 2:
            num_unique = 2
            
        # Generate the set
        data_set = np.random.randint(min_value, num_unique, size=set_size)
    
    # Calculate entropies
    h_min = min_entropy(data_set)
    h = shannon_entropy(data_set)
    
    results.append((h_min, h))

# Output results
print("(Min-Entropy, Shannon Entropy)")
for i, (h_min, h) in enumerate(results):
    print(f"Set {i+1}: ({h_min:.4f}, {h:.4f})")

# Output as ordered pairs only
print("\nOrdered pairs (h_min, h):")
print([(round(h_min, 4), round(h, 4)) for h_min, h in results])
