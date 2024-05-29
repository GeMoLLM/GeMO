import math
import matplotlib.pyplot as plt

def decaying_temperature(token_idx):
    if token_idx >= 50:
        return 1.0

    start_temp = 10.0
    end_temp = 1.5
    decay_period = 50

    relative_idx = token_idx / decay_period
    temp = start_temp * math.exp(-5 * relative_idx) + end_temp * (1 - math.exp(-5 * relative_idx))

    return temp

def linear_decay_temperature(token_idx):
    if token_idx >= 50:
        return 1.0
    start_temp = 10.0
    end_temp = 1.5
    decay_period = 50
    slope = (end_temp - start_temp) / decay_period
    temp = start_temp + slope * token_idx
    return temp

# Generate token indices and corresponding temperatures
token_indices = range(50)
temperatures = [decaying_temperature(idx) for idx in token_indices]
print(temperatures)
print(len(temperatures))

d = {i+1:temperatures[i] for i in range(50)}
print(d)

token_indices = range(50)
temperatures = [linear_decay_temperature(idx) for idx in token_indices]
print(temperatures)
print(len(temperatures))

d = {i+1:temperatures[i] for i in range(50)}
print(d)