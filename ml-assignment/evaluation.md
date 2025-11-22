# Evaluation

## Task 1: Trigram Language Model Design Choices

### Data Structure for N-Gram Counts
To store the Trigram counts, I utilized a Nested Dictionary structure using Python's `collections` library.
- Structure: `self.model = defaultdict(lambda: defaultdict(Counter))`
- Reasoning: 
    1.  Sparse Storage: A 3D array would be memory inefficient (mostly zeros). A dictionary only stores observed combinations.
    2.  O(1) Access: Accessing `model[w1][w2]` returns a `Counter` object containing all possible next words and their frequencies in constant time.
    3.  Code Cleanliness: `defaultdict` handles the initialization of missing keys automatically, removing the need for complex `if key in dict` checks during the training loop.

### Text Cleaning, Padding, and Unknown Words
- Tokenization: I implemented a regex-based tokenizer (`re.findall(r"\w+|[.!?]", text)`). This treats punctuation as separate tokens, allowing the model to learn sentence boundaries effectively.
- Padding: I padded sentences with double start tokens (`<START>`, `<START>`) and single end tokens (`<END>`). This is crucial for a Trigram model ($N=3$) so it can learn how sentences begin based on "empty" context.
- Unknown Words (<UNK>): To improve generalization, I implemented a frequency threshold. Words appearing fewer than 2 times in the training corpus are converted to `<UNK>`. This prevents the model from overfitting to unique names or typos and ensures it can handle unseen words during validation.

### Generation and Probabilistic Sampling
- Sampling Strategy: Instead of a "Greedy Search" (always picking the word with the highest count), I implemented Weighted Random Sampling.
- Implementation: I converted the raw counts in the `Counter` object to probabilities ($P = \frac{count}{total}$) and used `random.choices` to select the next word based on this distribution. This ensures the generated text retains variety and natural "randomness" while adhering to the statistical properties of the source text.
- Seeding: To generate valid text, the model selects a random existing bigram $(w_1, w_2)$ from the training data to use as the initial context seed.

---

## Task 2: Scaled Dot-Product Attention (NumPy)

### Mathematical Implementation
I implemented the standard Transformer attention formula using pure NumPy:
$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

### Numerical Stability (Softmax)
A critical engineering choice was made in the custom `softmax` function. I subtracted the maximum value from the logits before exponentiation: `exp(x - max(x))`.
- Reason: Exponential functions grow incredibly fast. Without this shift, large dot products would result in `inf` (Overflow) or `nan` errors. This trick ensures the largest value is 0, keeping the range safe for floating-point arithmetic.

### Masking Strategy
I implemented optional masking to handle padding in sequences.
- Implementation: `scaled_attention_logits += (mask == 0) * -1e9`
- Reason: By adding a large negative number (approx. negative infinity) to masked positions, the subsequent softmax operation drives the probability of these tokens to effectively zero ($e^{-10^9} \approx 0$). This ensures the model does not attend to padding tokens.
