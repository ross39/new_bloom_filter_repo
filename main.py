import xxhash
import math
import random
import string
random.seed(42)
# This is a comment
class BloomFilter:
    def __init__(self, size, hash_count):
        self.size = size
        self.hash_count = hash_count
        self. r = hash_count - math.floor(hash_count)
        self.bit_array = [0] * size
        self.witness = {}  # To store witness information

    def _hash(self, item, seed):
        return xxhash.xxh64(item, seed=seed).intdigest() % self.size

    def hash_to_float(self, input_string):
        hashed_value = xxhash.xxh64(input_string).intdigest()
        return hashed_value / (2**64 - 1)

    def add(self, item):
        if self.hash_to_float(item) < self.r:
            for i in range(math.floor(self.hash_count) + 1):
                index = self._hash(item, i)
                self.bit_array[index] = 1
        else:
            for i in range(math.floor(self.hash_count)):
                index = self._hash(item, i)
                self.bit_array[index] = 1

    def contains(self, item):
        if self.hash_to_float(item) < self.r:
            for i in range(math.floor(self.hash_count) + 1):
                index = self._hash(item, i)
                if self.bit_array[index] == 0:
                    return False
            return True
        else:
            for i in range(math.floor(self.hash_count)):
                index = self._hash(item, i)
                if self.bit_array[index] == 0:
                    return False
            return True

    @classmethod
    def get_size(cls, n, p):
        m = -(n * math.log(p)) / (math.log(2) ** 2)
        return int(m)

    @classmethod
    def get_hash_count(cls, m, n):
        k = (m / n) * math.log(2)
        return k

    def arithmetic_encode(self, input_string):
        # Convert the input string into a list of bits for easier manipulation
        bits = [int(c) for c in input_string]

        # Calculate frequency probabilities
        ones_count = sum(bits)
        total_bits = len(bits)
        prob_one = ones_count / total_bits if total_bits != 0 else 0.0
        prob_zero = 1.0 - prob_one

        # Initialize low and high for arithmetic coding
        low = 0.0
        high = 1.0

        for bit in bits:
            mid = (low + high) / 2
            if bit == 1:
                low = mid
            else:
                high = mid

        # The encoded value is the midpoint of the final interval
        return (low + high) / 2

    def arithmetic_decode(self, compressed_value, length):
        bits = []
        low = 0.0
        high = 1.0

        for _ in range(length):
            mid = (low + high) / 2
            if compressed_value > mid:
                bits.append('1')
                low = mid
            else:
                bits.append('0')
                high = mid

        return ''.join(bits)

    def compress(self):
        # Combine the bit_array and witness into a single string for compression
        combined_bitstring = ''.join(map(str, self.bit_array))

        # For simplicity, assuming the witness data is small enough to be part of metadata
        # In practice, integrate it with the main data
        compressed_data = self.arithmetic_encode(combined_bitstring)
        return compressed_data

    def decompress(self, compressed_value, size):
        # Decompress the combined bitstream
        reconstructed = self.arithmetic_decode(compressed_value, size)

        # Split into original bit_array and witness (if needed)
        # This is a simplified approach; actual implementation may vary based on how data was combined
        bit_length = len(self.bit_array)
        bit_array_reconstructed = reconstructed[:bit_length]
        # Witness reconstruction would need additional information

        return list(map(int, bit_array_reconstructed))

# Example usage:
bf = BloomFilter(1, 0.5)
bf.add("cat")
compressed = bf.compress()
print(f"Compressed value: {compressed}")
decompressed_bits = bf.decompress(compressed, 1)
print(f"Decompressed bits: {decompressed_bits}")

# Testing false positives
num_test_strings = 1500
fp = 0
for _ in range(num_test_strings):
    test_string = ''.join(random.choices(string.ascii_lowercase, k=10))
    if test_string != "cat" and bf.contains(test_string):
        fp += 1

print(f"False Positive Rate: {fp / num_test_strings}")
