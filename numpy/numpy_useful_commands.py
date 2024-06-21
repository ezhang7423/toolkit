# Here's an expanded version of the code with additional examples and explanations for each section:

# ```python
import numpy as np
import einops

###########
# Linear Algebra
###########

# * Inverse
A = np.array([[1, 2], [3, 4]])
A_inv = np.linalg.inv(A)
print("Inverse of A:")
print(A_inv)

# * Determinant
det_A = np.linalg.det(A)
print(f"Determinant of A: {det_A}")

# * Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)
print("Eigenvalues:")
print(eigenvalues)
print("Eigenvectors:")
print(eigenvectors)

# * Solve linear system
b = np.array([1, 2])
x = np.linalg.solve(A, b)
print("Solution to Ax = b:")
print(x)

# * Matrix multiplication
B = np.array([[5, 6], [7, 8]])
C = np.matmul(A, B)  # or A @ B
print("A * B:")
print(C)

# * Dot product
v1 = np.array([1, 2])
v2 = np.array([3, 4])
dot_product = np.dot(v1, v2)
print(f"Dot product of v1 and v2: {dot_product}")

# * Matrix rank
rank_A = np.linalg.matrix_rank(A)
print(f"Rank of A: {rank_A}")

# * Singular Value Decomposition (SVD)
U, s, Vt = np.linalg.svd(A)
print("SVD of A:")
print("U:", U)
print("s:", s)
print("V^T:", Vt)

# * Pseudo-inverse
A_pinv = np.linalg.pinv(A)
print("Pseudo-inverse of A:")
print(A_pinv)

#########
# Statistics
#########

data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# * Mean
mean = np.mean(data)
print(f"Mean: {mean}")

# * Median
median = np.median(data)
print(f"Median: {median}")

# * Standard deviation
std_dev = np.std(data)
print(f"Standard deviation: {std_dev}")

# * Variance
variance = np.var(data)
print(f"Variance: {variance}")

# * Correlation coefficient
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])
corr_coef = np.corrcoef(x, y)[0, 1]
print(f"Correlation coefficient: {corr_coef}")

# * Covariance
cov_matrix = np.cov(x, y)
print("Covariance matrix:")
print(cov_matrix)

# * Histogram
hist, bin_edges = np.histogram(data, bins=5)
print("Histogram:")
print("Counts:", hist)
print("Bin edges:", bin_edges)

# * Percentiles
percentiles = np.percentile(data, [25, 50, 75])
print("Percentiles (25th, 50th, 75th):")
print(percentiles)

# * Random numbers
uniform_random = np.random.rand(3, 3)  # uniform random numbers
normal_random = np.random.randn(3, 3)  # standard normal random numbers
integers_random = np.random.randint(0, 10, 5)  # 5 random integers between 0 and 10

print("Uniform random numbers:")
print(uniform_random)
print("Normal random numbers:")
print(normal_random)
print("Random integers:")
print(integers_random)

#########
# Reshaping and Array Manipulation
#########

# Create a sample array
arr = np.arange(24).reshape(2, 3, 4)

# * Reshape
reshaped = arr.reshape(6, 4)
print("Reshaped array:")
print(reshaped)

# * Transpose
transposed = arr.transpose()
print("Transposed array:")
print(transposed)

# * Flatten
flattened = arr.flatten()
print("Flattened array:")
print(flattened)

# * Expand dimensions
expanded = np.expand_dims(arr, axis=0)
print("Expanded array shape:", expanded.shape)

# * Squeeze
squeezed = np.squeeze(expanded)
print("Squeezed array shape:", squeezed.shape)

# * Concatenate
concat = np.concatenate([arr, arr], axis=0)
print("Concatenated array shape:", concat.shape)

# * Stack
stacked = np.stack([arr, arr], axis=0)
print("Stacked array shape:", stacked.shape)

# * Split
split = np.split(arr, 2, axis=0)
print("Split array shapes:", [s.shape for s in split])

# Using einops for more advanced reshaping
#* Rearrange dimensions
rearranged = einops.rearrange(arr, 'a b c -> b c a')
print("Rearranged array shape:", rearranged.shape)

#* Repeat elements
repeated = einops.repeat(arr, 'a b c -> (2 a) b c')
print("Repeated array shape:", repeated.shape)

#* Reduce dimensions
reduced = einops.reduce(arr, 'a b c -> b c', 'sum')
print("Reduced array shape:", reduced.shape)

#########
# Additional NumPy Operations
#########

# * Element-wise operations
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print("Element-wise addition:", a + b)
print("Element-wise multiplication:", a * b)

# * Broadcasting
c = np.array([[1, 2, 3], [4, 5, 6]])
d = np.array([1, 2, 3])
print("Broadcasting result:")
print(c + d)

# * Masked operations
mask = np.array([True, False, True])
masked_arr = np.ma.masked_array(a, mask=mask)
print("Masked array:", masked_arr)

# * Universal functions (ufuncs)
print("Sine of array:", np.sin(a))
print("Exponential of array:", np.exp(a))

# * Vectorization
def f(x, y):
    return x**2 + y**2

vectorized_f = np.vectorize(f)
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])
print("Vectorized function result:", vectorized_f(x, y))

# * Fancy indexing
indices = np.array([0, 2])
print("Fancy indexing result:", a[indices])

# * Boolean indexing
bool_mask = a > 1
print("Boolean indexing result:", a[bool_mask])

# * Set operations
set1 = np.array([1, 2, 3, 4, 5])
set2 = np.array([4, 5, 6, 7, 8])
print("Unique elements:", np.unique(np.concatenate((set1, set2))))
print("Intersection:", np.intersect1d(set1, set2))
print("Union:", np.union1d(set1, set2))

# * Polynomial operations
coeffs = [1, -2, 1]  # x^2 - 2x + 1
x = np.linspace(-2, 2, 100)
y = np.polyval(coeffs, x)
print("Polynomial roots:", np.roots(coeffs))

# * Fast Fourier Transform (FFT)
time = np.linspace(0, 10, 1000)
signal = np.sin(2 * np.pi * 10 * time) + 0.5 * np.sin(2 * np.pi * 20 * time)
fft_result = np.fft.fft(signal)
frequencies = np.fft.fftfreq(len(time), time[1] - time[0])
print("FFT result shape:", fft_result.shape)
print("Frequency array shape:", frequencies.shape)
# ```

# This expanded version includes:

# 1. More linear algebra operations (rank, SVD, pseudo-inverse)
# 2. Additional statistical functions (median, variance, percentiles)
# 3. Comprehensive reshaping and array manipulation examples, including einops
# 4. Element-wise operations and broadcasting
# 5. Masked arrays
# 6. Universal functions (ufuncs)
# 7. Vectorization
# 8. Fancy and boolean indexing
# 9. Set operations
# 10. Polynomial operations
# 11. Fast Fourier Transform (FFT)

# This code provides a broad overview of NumPy's capabilities in linear algebra, statistics, array manipulation, and various mathematical operations.

# Would you like me to explain or break down any part of this code?
