import xxhash
import math
import random
from pybloomfilter import BloomFilter
from termcolor import colored


class MaliciousUrlFilter:
    def __init__(self, size, hash_count):
        self.size = size
        self.hash_count = hash_count
        self.bit_array = [0] * size

    def _hash(self, item, seed):
        return xxhash.xxh128(item, seed).intdigest() % self.size

    def add(self, item):
        for i in range(math.floor(self.hash_count) + 1):
            index = self._hash(item, i)
            self.bit_array[index] = 1

    def contains(self, item, p):
        if random.random() < p:
            for i in range(math.floor(self.hash_count) + 1):
                index = self._hash(item, i)
                if self.bit_array[index] == 0:
                    return False
            return True
        else:
            if math.floor(self.hash_count) == 0:
                return False
            for i in range(math.floor(self.hash_count)):
                index = self._hash(item, i)
                if self.bit_array[index] == 0:
                    return False
            return True

# Use one filter for conditional probability
# We may miss true positives but we trade that for better false positive rate
#
# If a malicious url is more likely to possess a propety than a non-malicious url then we
# can reduce the FPR
#
# Specifically, it is assumed that the majority of malicious URLs do satisfy P and
# the majority of non-malicious URLs do not satisfy P, corresponding to b ≤ 1/2 ≤ a
#
# When performing a Bloom filter test (bft), the number of hash function calls is
# modified according to whether the URL satisfies property P
#
# If the URL satisfies P, the number of hash function calls is floor(k) + 1(url is likely malicious so lets check more)
# If the URL does not satisfy P, the number of hash function calls is floor(k) (url is likely non-malicious so lets check less)

URLs = [
    'fake.com',
    'malicious.com',
    'malicious.net',
    'malicious.org',
    'scam.ie',
    'https:malware.fr',
    'https:malware.co.uk',
    'amazon.com',
    'google.com',
    'facebook.com',
    'youtube.com',
    'twitter.com',
    'instagram.com',
    'linkedin.com',
    'reddit.com',
    'pinterest.com',
    'tumblr.com',
]
# I calculated my values using https://hur.st/bloomfilter/?n=17&p=0.1&m=&k=
# m = 82
# k = 3.34
print('Size of n', len(URLs))
MF = MaliciousUrlFilter(82, 3.34)
print("value of k for conditional", MF.hash_count)
print("value of m for conditional", MF.size)
for url in URLs:
    MF.add(url)

TEST_URLs = {
    'fake.com': 1,
    'malicious.com': 1,
    'malicious.net': 1,
    'malicious.org': 1,
    'scam.ie': 1,
    'https:malware.fr': 1,
    'https:malware.co.uk': 1,
    'amazon.com': 1,
    'notsure.com': 0.5,
    'letstest.ie': 0.1,
    'church': 0.01,
    'definitelynotfake.com': 0.9,
}

print(colored("Testing URLS in conditional filter", "red", "on_white"))
for url, p in TEST_URLs.items():
    print(f'{url} is malicious: {MF.contains(url, p)}')


# The filter performs with a lower fpr rate than a standard bloom filter as it can use rational k values
# This allows it to adjust k based on the probability of a URL being malicious
# Multiple standard bloom filters would be needed to achieve the same result

# Obviously this depends on us having accurate probabilities for the URLs. If I add a URL to the filter and then test
# it with a different probability, the filter will not work as expected. For example
# MF.add('notsure.com') -> Its assumed this url has p = 1 as we added it to the filter
# MF.contains('notsure.com', 0.1) -> Will likely return false as it may use floor(k) instead of floor(k) + 1


# Compare to a standard bloom filter
# I set the fpr rate to be what the formula calculates so that I can compare the filter sizes and k values
# for the same fpr rate
bf = BloomFilter(17, 0.1)
# Note the integer value of k differs from the rational value of k
print("value of k for standard", bf.num_hashes)
# Note the size of the bit array is much bigger
print("value of m for standard", bf.num_bits)
for url in URLs:
    bf.add(url)



# We cannot use p here so just assume all URLs are equally likely to be malicious
# This will result in a higher fpr rate
print(colored("Testing URLS in standard filter", "red", "on_white"))
for url in TEST_URLs.keys():
    print(f'{url} is malicious: {bf.add(url)}')


# It must be said that it appears 128 is the smallest size we can use for pybloomfiltermmap3 as it will resize
# when the filter density reaches a certain value. So I don't think this is truly a fair comparison but i hope it
# illustrates the point
