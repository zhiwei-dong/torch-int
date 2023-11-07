import torch
import time

# Define the tensors
values = torch.tensor([1, 2, 3]).cuda()
repeats = torch.tensor([2, 1, 3]).cuda()

# Create a tensor cumulative_indices with the total number of elements required
cumulative_indices = torch.zeros(repeats.sum(), dtype=torch.long, device=repeats.device)

# The first index is always 0
cumulative_indices[0] = 0

# Calculate the start indices for each new value
cumulative_sums = repeats.cumsum(0)
start_indices = torch.zeros_like(repeats)
start_indices[1:] = cumulative_sums[:-1]  # Set the start index for each group

# Scatter the start_indices into the cumulative_indices tensor
cumulative_indices.scatter_(0, start_indices, 1)

# The cumulative sum of cumulative_indices now gives us the indices we want
index_tensor = cumulative_indices.cumsum(0) - 1

# Time the torch.repeat_interleave function
start_time = time.time()
for i in range(100):
    result_repeat_interleave = torch.repeat_interleave(values, repeats)
time_repeat_interleave = time.time() - start_time

# Time the corrected efficient method
start_time = time.time()

for i in range(100):
    # Index into values using index_tensor to repeat the elements
    result_efficient = values[index_tensor]

time_efficient = time.time() - start_time

# Check if the results are the same
results_match_efficient = torch.equal(result_repeat_interleave, result_efficient)

# Display the results
print("Results match:", results_match_efficient)
print("Result from torch.repeat_interleave:", result_repeat_interleave)
print("Result from the efficient method:", result_efficient)
print("Time for torch.repeat_interleave:", time_repeat_interleave)
print("Time for the efficient method:", time_efficient)