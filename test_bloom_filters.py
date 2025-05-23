import random
import string
import numpy as np
import matplotlib.pyplot as plt
import math
from rational_bloom_filter import StandardBloomFilter, RationalBloomFilter

def generate_random_strings(n, length=10):
    """Generate n random strings of specified length."""
    return [''.join(random.choices(string.ascii_lowercase, k=length)) for _ in range(n)]

def test_small_example():
    """Test with a small example to visualize the difference."""
    print("\n=== Small Example Test ===")
    
    # Parameters: very small m and n to make the difference obvious
    m, n = 10, 5
    
    # Calculate optimal k* for the given m and n
    k_star = RationalBloomFilter.get_optimal_hash_count(m, n)
    k_std_floor = math.floor(k_star)
    k_std_ceil = math.ceil(k_star)
    
    print(f"Parameters: m={m}, n={n}")
    print(f"Optimal k*: {k_star:.4f}")
    print(f"Standard options: floor(k*)={k_std_floor} or ceil(k*)={k_std_ceil}")
    
    # Create filters
    std_filter_floor = StandardBloomFilter(m, k_std_floor)
    std_filter_ceil = StandardBloomFilter(m, k_std_ceil)
    rational_filter = RationalBloomFilter(m, k_star)
    
    # Generate elements to insert
    elements = generate_random_strings(n)
    
    # Insert elements
    for element in elements:
        std_filter_floor.add(element)
        std_filter_ceil.add(element)
        rational_filter.add(element)
    
    # Print the bit arrays
    print("\nBit Arrays:")
    print(f"Standard (k={k_std_floor}): {std_filter_floor.bit_array}")
    print(f"Standard (k={k_std_ceil}): {std_filter_ceil.bit_array}")
    print(f"Rational (k*={k_star:.4f}): {rational_filter.bit_array}")
    
    # Count bits set
    bits_floor = sum(std_filter_floor.bit_array)
    bits_ceil = sum(std_filter_ceil.bit_array)
    bits_rational = sum(rational_filter.bit_array)
    
    print(f"\nBits set in Standard (k={k_std_floor}): {bits_floor}/{m}")
    print(f"Bits set in Standard (k={k_std_ceil}): {bits_ceil}/{m}")
    print(f"Bits set in Rational (k*={k_star:.4f}): {bits_rational}/{m}")
    
    # Test with new elements
    num_test = 100
    test_elements = generate_random_strings(num_test)
    
    fp_floor = sum(1 for e in test_elements if std_filter_floor.contains(e) and e not in elements)
    fp_ceil = sum(1 for e in test_elements if std_filter_ceil.contains(e) and e not in elements)
    fp_rational = sum(1 for e in test_elements if rational_filter.contains(e) and e not in elements)
    
    print(f"\nFalse positives with Standard (k={k_std_floor}): {fp_floor}/{num_test} = {fp_floor/num_test:.4f}")
    print(f"False positives with Standard (k={k_std_ceil}): {fp_ceil}/{num_test} = {fp_ceil/num_test:.4f}")
    print(f"False positives with Rational (k*={k_star:.4f}): {fp_rational}/{num_test} = {fp_rational/num_test:.4f}")

def compare_varying_m_n():
    """Compare filters with varying m/n ratio."""
    print("\n=== Varying m/n Ratio Test ===")
    
    # Test with different m/n ratios
    n = 100  # Fixed number of elements
    m_values = [int(n * ratio) for ratio in np.linspace(2, 20, 10)]  # Different m/n ratios
    
    std_fprs = []
    rational_fprs = []
    k_stars = []
    
    for m in m_values:
        # Calculate optimal k* for this m and n
        k_star = RationalBloomFilter.get_optimal_hash_count(m, n)
        k_std = StandardBloomFilter.get_optimal_hash_count(m, n)
        k_stars.append(k_star)
        
        # Create filters
        std_filter = StandardBloomFilter(m, k_std)
        rational_filter = RationalBloomFilter(m, k_star)
        
        # Generate elements and test elements
        elements = set(generate_random_strings(n))
        test_elements = generate_random_strings(10000)  # Large number for accurate FPR
        
        # Insert elements
        for element in elements:
            std_filter.add(element)
            rational_filter.add(element)
        
        # Measure false positive rates
        fp_std = sum(1 for e in test_elements if std_filter.contains(e) and e not in elements)
        fp_rational = sum(1 for e in test_elements if rational_filter.contains(e) and e not in elements)
        
        std_fprs.append(fp_std / len(test_elements))
        rational_fprs.append(fp_rational / len(test_elements))
        
        print(f"m={m}, m/n={m/n:.2f}, k*={k_star:.4f}, k_std={k_std}")
        print(f"  Standard FPR: {std_fprs[-1]:.6f}")
        print(f"  Rational FPR: {rational_fprs[-1]:.6f}")
        if std_fprs[-1] > 0:
            improvement = (std_fprs[-1] - rational_fprs[-1]) / std_fprs[-1] * 100
            print(f"  Improvement: {improvement:.2f}%")
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot([m/n for m in m_values], std_fprs, 'o-', label='Standard Bloom Filter')
    plt.plot([m/n for m in m_values], rational_fprs, 's-', label='Rational Bloom Filter')
    plt.xlabel('m/n Ratio')
    plt.ylabel('False Positive Rate')
    plt.title('False Positive Rate vs m/n Ratio')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    improvements = [(std_fprs[i] - rational_fprs[i]) / std_fprs[i] * 100 if std_fprs[i] > 0 else 0 
                   for i in range(len(std_fprs))]
    plt.bar([m/n for m in m_values], improvements)
    plt.xlabel('m/n Ratio')
    plt.ylabel('Improvement (%)')
    plt.title('Improvement of Rational over Standard Bloom Filter')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('bloom_filter_varying_mn.png')
    print("Results saved to bloom_filter_varying_mn.png")

def test_theoretical_vs_empirical():
    """Compare theoretical vs empirical false positive rates."""
    print("\n=== Theoretical vs Empirical False Positive Rates ===")
    
    # Parameters
    m, n = 100, 10
    k_star = RationalBloomFilter.get_optimal_hash_count(m, n)
    k_std = StandardBloomFilter.get_optimal_hash_count(m, n)
    
    # Theoretical false positive rates
    # For standard BF: (1 - e^(-k*n/m))^k
    # For rational BF with k* = floor(k) + p: (1 - e^(-floor(k)*n/m))^floor(k) * (1 - e^(-n/m))^p
    p = k_star - math.floor(k_star)
    theoretical_std = (1 - np.exp(-k_std * n / m)) ** k_std
    theoretical_rational_simple = (1 - np.exp(-k_star * n / m)) ** k_star
    theoretical_rational_exact = (1 - np.exp(-math.floor(k_star) * n / m)) ** math.floor(k_star) * \
                               (1 - np.exp(-n / m)) ** p
    
    print(f"Parameters: m={m}, n={n}, k*={k_star:.4f}, k_std={k_std}")
    print(f"Theoretical FPR (Standard): {theoretical_std:.6f}")
    print(f"Theoretical FPR (Rational, simple approximation): {theoretical_rational_simple:.6f}")
    print(f"Theoretical FPR (Rational, exact formula): {theoretical_rational_exact:.6f}")
    
    # Empirical measurement with large number of trials
    num_trials = 10
    std_fprs = []
    rational_fprs = []
    
    for trial in range(num_trials):
        # Create filters
        std_filter = StandardBloomFilter(m, k_std)
        rational_filter = RationalBloomFilter(m, k_star)
        
        # Generate elements and test elements
        elements = set(generate_random_strings(n))
        test_elements = generate_random_strings(100000)  # Very large for accurate FPR
        
        # Insert elements
        for element in elements:
            std_filter.add(element)
            rational_filter.add(element)
        
        # Measure false positive rates
        fp_std = sum(1 for e in test_elements if std_filter.contains(e) and e not in elements)
        fp_rational = sum(1 for e in test_elements if rational_filter.contains(e) and e not in elements)
        
        std_fprs.append(fp_std / len(test_elements))
        rational_fprs.append(fp_rational / len(test_elements))
    
    empirical_std = np.mean(std_fprs)
    empirical_rational = np.mean(rational_fprs)
    
    print(f"Empirical FPR (Standard): {empirical_std:.6f}")
    print(f"Empirical FPR (Rational): {empirical_rational:.6f}")
    
    # Compare with theoretical predictions
    std_error = abs(empirical_std - theoretical_std) / theoretical_std * 100
    rational_error_simple = abs(empirical_rational - theoretical_rational_simple) / theoretical_rational_simple * 100
    rational_error_exact = abs(empirical_rational - theoretical_rational_exact) / theoretical_rational_exact * 100
    
    print(f"Standard BF - Theoretical vs Empirical error: {std_error:.2f}%")
    print(f"Rational BF - Simple approximation error: {rational_error_simple:.2f}%")
    print(f"Rational BF - Exact formula error: {rational_error_exact:.2f}%")

if __name__ == "__main__":
    random.seed(42)
    
    print("Rational Bloom Filter Tests")
    print("==========================")
    
    test_small_example()
    compare_varying_m_n()
    test_theoretical_vs_empirical() 