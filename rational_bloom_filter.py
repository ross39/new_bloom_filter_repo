import xxhash
import math
import random
import string
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Set, Tuple, Union

class StandardBloomFilter:
    """
    Implementation of a standard Bloom filter where k must be an integer.
    """
    def __init__(self, m: int, k: int):
        """
        Initialize a standard Bloom filter.
        
        Args:
            m: Size of the bit array
            k: Number of hash functions (must be an integer)
        """
        self.size = m
        self.hash_count = int(k)  # Ensure k is an integer
        self.bit_array = [0] * m
    
    def _hash(self, item: str, seed: int) -> int:
        """Generate a hash value for the given item and seed."""
        return xxhash.xxh64(str(item), seed=seed).intdigest() % self.size
    
    def add(self, item: str) -> None:
        """Add an item to the Bloom filter."""
        for i in range(self.hash_count):
            index = self._hash(item, i)
            self.bit_array[index] = 1
    
    def contains(self, item: str) -> bool:
        """Check if an item might be in the Bloom filter."""
        for i in range(self.hash_count):
            index = self._hash(item, i)
            if self.bit_array[index] == 0:
                return False
        return True
    
    @staticmethod
    def get_optimal_size(n: int, p: float) -> int:
        """
        Calculate the optimal bit array size for n elements with false positive rate p.
        
        Args:
            n: Number of elements to insert
            p: Desired false positive rate
            
        Returns:
            Optimal size m of the bit array
        """
        m = -(n * math.log(p)) / (math.log(2) ** 2)
        return int(math.ceil(m))
    
    @staticmethod
    def get_optimal_hash_count(m: int, n: int) -> int:
        """
        Calculate the optimal number of hash functions for a Bloom filter.
        
        Args:
            m: Size of the bit array
            n: Number of elements to insert
            
        Returns:
            Optimal number of hash functions k (rounded to an integer)
        """
        k = (m / n) * math.log(2)
        return max(1, int(round(k)))  # Ensure k ≥ 1


class RationalBloomFilter:
    """
    Implementation of a Rational Bloom filter as described in
    "Extending the Applicability of Bloom Filters by Relaxing their Parameter Constraints"
    by Paul Walther et al.
    
    The Rational Bloom filter allows for a non-integer number of hash functions (k*),
    which is achieved by probabilistically applying an additional hash function
    beyond the floor(k*) deterministic hash functions.
    """
    def __init__(self, m: int, k_star: float):
        """
        Initialize a Rational Bloom filter.
        
        Args:
            m: Size of the bit array
            k_star: Optimal (rational) number of hash functions
        """
        self.size = m
        self.k_star = k_star
        self.floor_k = math.floor(k_star)
        self.ceil_k = math.ceil(k_star)
        self.p_activation = k_star - self.floor_k  # Fractional part used as probability
        self.bit_array = [0] * m
        
        # Create two base hash functions for the double hashing technique
        self.h1_seed = 0
        self.h2_seed = 1
    
    def _get_hash_indices(self, item: str, i: int) -> int:
        """
        Implement the double hashing technique to generate hash indices.
        This is more efficient than having k completely independent hash functions.
        
        Args:
            item: The item to hash
            i: The index of the hash function (0 to ceil_k-1)
            
        Returns:
            A hash index in the range [0, m-1]
        """
        h1 = xxhash.xxh64(str(item), seed=self.h1_seed).intdigest()
        h2 = xxhash.xxh64(str(item), seed=self.h2_seed).intdigest()
        
        # Use the double hashing technique: (h1(x) + i * h2(x)) % m
        return (h1 + i * h2) % self.size
    
    def _determine_activation(self, item: str) -> bool:
        """
        Deterministically decide whether to apply the additional hash function
        for the given item based on the fractional part of k*.
        
        Args:
            item: The item to check
            
        Returns:
            True if the additional hash function should be applied, False otherwise
        """
        # Use a hash of the item to create a deterministic decision
        # This ensures the same decision is made for the same item during both add and contains
        hash_value = xxhash.xxh64(str(item), seed=self.ceil_k).intdigest()
        normalized_value = hash_value / (2**64 - 1)  # Convert to [0,1)
        
        return normalized_value < self.p_activation
    
    def add(self, item: str) -> None:
        """
        Add an item to the Rational Bloom filter.
        
        For each item, we:
        1. Always apply the first floor(k*) hash functions
        2. Probabilistically apply the ceiling hash function based on p_activation
        """
        # Always apply the floor(k*) hash functions deterministically
        for i in range(self.floor_k):
            index = self._get_hash_indices(item, i)
            self.bit_array[index] = 1
        
        # Probabilistically apply the additional hash function
        # if the activation probability test passes
        if self._determine_activation(item):
            index = self._get_hash_indices(item, self.floor_k)
            self.bit_array[index] = 1
    
    def contains(self, item: str) -> bool:
        """
        Check if an item might be in the Rational Bloom filter.
        
        According to the paper, we must:
        1. Check all deterministic hash functions (floor(k*))
        2. Check the probabilistic hash function ONLY if it would have been
           activated during insertion for this specific item
        
        This preserves the "no false negatives" property of Bloom filters.
        """
        # Check the deterministic hash functions (floor(k*))
        for i in range(self.floor_k):
            index = self._get_hash_indices(item, i)
            if self.bit_array[index] == 0:
                return False
        
        # Check the probabilistic hash function only if it would have been
        # activated during insertion for this specific item
        if self._determine_activation(item):
            index = self._get_hash_indices(item, self.floor_k)
            if self.bit_array[index] == 0:
                return False
        
        return True
    
    @staticmethod
    def get_optimal_size(n: int, p: float) -> int:
        """
        Calculate the optimal bit array size for n elements with false positive rate p.
        
        Args:
            n: Number of elements to insert
            p: Desired false positive rate
            
        Returns:
            Optimal size m of the bit array
        """
        m = -(n * math.log(p)) / (math.log(2) ** 2)
        return int(math.ceil(m))
    
    @staticmethod
    def get_optimal_hash_count(m: int, n: int) -> float:
        """
        Calculate the optimal (rational) number of hash functions k* for a Bloom filter.
        
        The formula is: k* = (m/n) * ln(2)
        
        Args:
            m: Size of the bit array
            n: Number of elements to insert
            
        Returns:
            Optimal number of hash functions k* (a rational number)
        """
        k_star = (m / n) * math.log(2)
        return max(0.1, k_star)  # Ensure k* is positive


def generate_random_strings(n: int, length: int = 10) -> List[str]:
    """Generate n random strings of specified length."""
    return [''.join(random.choices(string.ascii_lowercase, k=length)) for _ in range(n)]


def measure_false_positive_rate(bloom_filter: Union[StandardBloomFilter, RationalBloomFilter], 
                               true_elements: Set[str], 
                               test_elements: List[str]) -> float:
    """
    Measure the false positive rate of a Bloom filter.
    
    Args:
        bloom_filter: The Bloom filter to test
        true_elements: Set of elements that were actually inserted
        test_elements: List of elements to test (should be different from true_elements)
        
    Returns:
        False positive rate (proportion of false positives)
    """
    false_positives = 0
    for element in test_elements:
        if element not in true_elements and bloom_filter.contains(element):
            false_positives += 1
    
    return false_positives / len(test_elements)


def compare_filters(m: int, n: int, num_test_elements: int = 10000) -> Tuple[float, float]:
    """
    Compare the performance of Standard and Rational Bloom filters.
    
    Args:
        m: Size of the bit array
        n: Number of elements to insert
        num_test_elements: Number of elements to test for false positives
        
    Returns:
        Tuple of (standard_fpr, rational_fpr)
    """
    # Calculate optimal k* for the given m and n
    k_star = RationalBloomFilter.get_optimal_hash_count(m, n)
    k_std = StandardBloomFilter.get_optimal_hash_count(m, n)
    
    # Create both filters
    std_filter = StandardBloomFilter(m, k_std)
    rational_filter = RationalBloomFilter(m, k_star)
    
    # Generate true elements (to insert) and test elements (to check false positives)
    true_elements = set(generate_random_strings(n))
    
    # Generate test elements that are guaranteed not to be in the true elements
    test_elements = []
    while len(test_elements) < num_test_elements:
        element = ''.join(random.choices(string.ascii_lowercase, k=10))
        if element not in true_elements:
            test_elements.append(element)
    
    # Insert true elements into both filters
    for element in true_elements:
        std_filter.add(element)
        rational_filter.add(element)
    
    # Measure false positive rates
    std_fpr = measure_false_positive_rate(std_filter, true_elements, test_elements)
    rational_fpr = measure_false_positive_rate(rational_filter, true_elements, test_elements)
    
    return std_fpr, rational_fpr


def run_experiment_varying_k(m: int, n: int, k_values: List[float], num_test_elements: int = 10000) -> Tuple[List[float], List[float]]:
    """
    Run an experiment with various k values to find the optimal k.
    
    Args:
        m: Size of the bit array
        n: Number of elements to insert
        k_values: List of k values to test
        num_test_elements: Number of elements to test for false positives
        
    Returns:
        Tuple of (standard_fprs, rational_fprs)
    """
    # Generate true elements (to insert) and test elements (to check false positives)
    true_elements = set(generate_random_strings(n))
    
    # Generate test elements that are guaranteed not to be in the true elements
    test_elements = []
    while len(test_elements) < num_test_elements:
        element = ''.join(random.choices(string.ascii_lowercase, k=10))
        if element not in true_elements:
            test_elements.append(element)
    
    standard_fprs = []
    rational_fprs = []
    
    for k in k_values:
        # Create filters
        std_filter = StandardBloomFilter(m, int(round(k)))
        rational_filter = RationalBloomFilter(m, k)
        
        # Insert true elements
        for element in true_elements:
            std_filter.add(element)
            rational_filter.add(element)
        
        # Measure false positive rates
        std_fpr = measure_false_positive_rate(std_filter, true_elements, test_elements)
        rational_fpr = measure_false_positive_rate(rational_filter, true_elements, test_elements)
        
        standard_fprs.append(std_fpr)
        rational_fprs.append(rational_fpr)
    
    return standard_fprs, rational_fprs


def run_theoretical_comparison(m: int, n: int, k_values: List[float]) -> Tuple[List[float], List[float]]:
    """
    Calculate theoretical false positive rates for standard and rational Bloom filters.
    
    For standard filters with integer k: p = (1 - e^(-kn/m))^k
    For rational filters with rational k*: p = (1 - e^(-k*n/m))^floor(k*) * (1 - e^(-k*n/m) * p_activation)
    
    Args:
        m: Size of the bit array
        n: Number of elements to insert
        k_values: List of k values to calculate theoretical FPR for
        
    Returns:
        Tuple of (standard_theoretical_fprs, rational_theoretical_fprs)
    """
    standard_theoretical_fprs = []
    rational_theoretical_fprs = []
    
    for k in k_values:
        k_int = int(round(k))
        k_floor = math.floor(k)
        p_activation = k - k_floor
        
        # Standard Bloom filter theoretical FPR
        fill_ratio = 1 - math.exp(-k_int * n / m)
        std_fpr = fill_ratio ** k_int
        
        # Rational Bloom filter theoretical FPR
        fill_ratio_rational = 1 - math.exp(-k * n / m)
        rational_fpr = fill_ratio_rational ** k_floor
        if p_activation > 0:
            rational_fpr *= (1 - (1 - fill_ratio_rational) * p_activation)
        
        standard_theoretical_fprs.append(std_fpr)
        rational_theoretical_fprs.append(rational_fpr)
    
    return standard_theoretical_fprs, rational_theoretical_fprs


def main():
    # Set random seed for reproducibility
    random.seed(42)
    
    print("Comparing Standard and Rational Bloom Filters")
    print("=============================================")
    
    # Example 1: Simple comparison with fixed parameters
    m, n = 10, 50  # Using a larger size for more meaningful results
    k_star = RationalBloomFilter.get_optimal_hash_count(m, n)
    k_std = StandardBloomFilter.get_optimal_hash_count(m, n)
    
    print(f"Parameters: m={m}, n={n}")
    print(f"Optimal k*: {k_star:.4f}")
    print(f"Standard Bloom Filter using k={k_std}")
    print(f"Rational Bloom Filter using k*={k_star:.4f}")
    
    std_fpr, rational_fpr = compare_filters(m, n, num_test_elements=10000)
    
    print(f"Standard Bloom Filter FPR:   {std_fpr:.6f}")
    print(f"Rational Bloom Filter FPR:   {rational_fpr:.6f}")
    if std_fpr > 0:
        improvement = (std_fpr - rational_fpr) / std_fpr * 100
        print(f"Improvement: {improvement:.2f}%")
    
    # Example 2: Vary k to see the effect on FPR
    print("\nRunning experiment with varying k values...")
    
    # Test k values around the optimal k*
    k_min = max(0.1, k_star - 1.5)
    k_max = k_star + 1.5
    k_values = np.linspace(k_min, k_max, 30)
    
    std_fprs, rational_fprs = run_experiment_varying_k(m, n, k_values, num_test_elements=5000)
    
    # Also calculate theoretical FPRs
    std_theory_fprs, rational_theory_fprs = run_theoretical_comparison(m, n, k_values)
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    
    # Plot experimental results
    plt.plot(k_values, std_fprs, 'o-', label='Standard Bloom Filter (Experimental)', color='blue', alpha=0.7)
    plt.plot(k_values, rational_fprs, 's-', label='Rational Bloom Filter (Experimental)', color='green', alpha=0.7)
    
    # Plot theoretical results
    plt.plot(k_values, std_theory_fprs, '--', label='Standard Bloom Filter (Theoretical)', color='blue', alpha=0.4)
    plt.plot(k_values, rational_theory_fprs, '--', label='Rational Bloom Filter (Theoretical)', color='green', alpha=0.4)
    
    # Mark the optimal k*
    plt.axvline(x=k_star, color='r', linestyle='--', label=f'Optimal k*={k_star:.4f}')
    
    # Mark integer k values
    for i in range(int(k_min), int(k_max) + 1):
        plt.axvline(x=i, color='gray', linestyle=':', alpha=0.5)
    
    plt.xlabel('Number of Hash Functions (k)')
    plt.ylabel('False Positive Rate')
    plt.title('Comparison of Standard vs Rational Bloom Filter')
    plt.legend()
    plt.grid(True)
    plt.savefig('bloom_filter_comparison.png')
    
    print(f"Optimal k* = {k_star:.4f}")
    print("Results saved to bloom_filter_comparison.png")
    
    # Example 3: Compare performance with varying array sizes
    print("\nComparing performance with varying array sizes (m)...")
    
    m_values = [50, 100, 150, 200, 250, 300]
    n = 50  # Fixed number of elements
    
    std_fprs = []
    rational_fprs = []
    
    for m in m_values:
        k_star = RationalBloomFilter.get_optimal_hash_count(m, n)
        k_std = StandardBloomFilter.get_optimal_hash_count(m, n)
        
        std_filter = StandardBloomFilter(m, k_std)
        rational_filter = RationalBloomFilter(m, k_star)
        
        # Generate true elements and test elements
        true_elements = set(generate_random_strings(n))
        test_elements = []
        while len(test_elements) < 5000:
            element = ''.join(random.choices(string.ascii_lowercase, k=10))
            if element not in true_elements:
                test_elements.append(element)
        
        # Insert elements
        for element in true_elements:
            std_filter.add(element)
            rational_filter.add(element)
        
        # Measure FPRs
        std_fpr = measure_false_positive_rate(std_filter, true_elements, test_elements)
        rational_fpr = measure_false_positive_rate(rational_filter, true_elements, test_elements)
        
        std_fprs.append(std_fpr)
        rational_fprs.append(rational_fpr)
        
        print(f"m={m}, k*={k_star:.4f}, k_std={k_std}")
        print(f"  Standard FPR: {std_fpr:.6f}")
        print(f"  Rational FPR: {rational_fpr:.6f}")
        if std_fpr > 0:
            improvement = (std_fpr - rational_fpr) / std_fpr * 100
            print(f"  Improvement: {improvement:.2f}%")
    
    # Plot the results for varying m
    plt.figure(figsize=(10, 6))
    plt.plot(m_values, std_fprs, 'o-', label='Standard Bloom Filter')
    plt.plot(m_values, rational_fprs, 's-', label='Rational Bloom Filter')
    plt.xlabel('Bit Array Size (m)')
    plt.ylabel('False Positive Rate')
    plt.title('Effect of Array Size on False Positive Rate')
    plt.legend()
    plt.grid(True)
    plt.savefig('bloom_filter_size_comparison.png')
    print("Results saved to bloom_filter_size_comparison.png")


if __name__ == "__main__":
    main() 