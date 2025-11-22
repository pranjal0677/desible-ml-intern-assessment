import numpy as np

def softmax(x):
    """
    Compute softmax values for each set of scores in x.
    
    Engineering Note:
    We subtract np.max(x) for numerical stability. 
    If x contains large values (e.g., 1000), exp(x) returns infinity (overflow).
    Subtracting the max shifts values to a negative range (0 and below), 
    ensuring exp(x) is between 0 and 1.
    """
    # axis=-1 applies softmax to the last dimension (the embedding dimension)
    # keepdims=True ensures shapes remain compatible for broadcasting
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Computes the Scaled Dot-Product Attention.
    
    Formula: Attention(Q, K, V) = softmax( (Q @ K.T) / sqrt(d_k) ) @ V
    
    Args:
        Q: Queries numpy array of shape (..., seq_len_q, depth)
        K: Keys numpy array of shape (..., seq_len_k, depth)
        V: Values numpy array of shape (..., seq_len_v, depth_v)
        mask: Optional numpy array. 
              0 indicates a position to mask out (ignore), 
              1 indicates a position to keep.
              Shape should be broadcastable to (..., seq_len_q, seq_len_k).
              
    Returns:
        output: The weighted sum of values (Context Vector).
        attention_weights: The weights after softmax.
    """
    
    # 1. Matrix Multiply Q and K
    # We need the dot product of every query with every key.
    # If Q is (Batch, Seq, Depth) and K is (Batch, Seq, Depth),
    # we need to transpose K to (Batch, Depth, Seq) to get (Batch, Seq, Seq).
    # swapaxes(-1, -2) is a robust way to transpose the last two dimensions.
    matmul_qk = np.matmul(Q, K.swapaxes(-1, -2))
    
    # 2. Scale
    # We divide by the square root of the depth (d_k) to prevent the dot products
    # from growing too large, which would push softmax into regions with tiny gradients.
    d_k = Q.shape[-1]
    scaled_attention_logits = matmul_qk / np.sqrt(d_k)
    
    # 3. Apply Mask (Optional)
    if mask is not None:
        # We assume the mask uses 0 for padding (ignore) and 1 for content (keep).
        # We add a very large negative number (-1e9) to the logits where the mask is 0.
        # When we apply softmax, exp(-1e9) becomes effectively 0.
        
        # Note: We use += (mask == 0) * -1e9 to handle boolean or integer masks safely
        scaled_attention_logits += (mask == 0) * -1e9
        
    # 4. Softmax
    # This converts the scores into a probability distribution (summing to 1)
    attention_weights = softmax(scaled_attention_logits)
    
    # 5. Multiply by Values
    # We compute the weighted sum of the values using the attention weights.
    # (Batch, Seq, Seq) @ (Batch, Seq, Depth) -> (Batch, Seq, Depth)
    output = np.matmul(attention_weights, V)
    
    return output, attention_weights

def run_demo():
    print("=== Task 2: Scaled Dot-Product Attention Demo ===\n")
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Dimensions
    # Batch Size = 1
    # Sequence Length = 3 (e.g., "The cat sat")
    # Embedding Dimension = 4
    batch_size = 1
    seq_len = 3
    d_model = 4
    
    print(f"Dimensions: Batch={batch_size}, Seq={seq_len}, Depth={d_model}")

    # Create random Q, K, V matrices
    Q = np.random.rand(batch_size, seq_len, d_model)
    K = np.random.rand(batch_size, seq_len, d_model)
    V = np.random.rand(batch_size, seq_len, d_model)

    print("\n--- Test 1: Standard Attention (No Mask) ---")
    output, weights = scaled_dot_product_attention(Q, K, V)
    
    print("Attention Weights shape:", weights.shape)
    print("Context Output shape:", output.shape)
    print("Weights Row 1 Sum (Should be ~1.0):", np.sum(weights[0, 0, :]))
    print("\nCalculated Weights:\n", np.round(weights, 4))

    print("\n" + "="*40 + "\n")

    print("--- Test 2: Masked Attention ---")
    # Let's pretend the 3rd token is padding.
    # Mask: 1 means keep, 0 means ignore.
    # Shape: (Batch, Seq_Q, Seq_K)
    mask = np.array([[[1, 1, 0],  # Query 1 can see Key 1, 2 (not 3)
                      [1, 1, 0],  # Query 2 can see Key 1, 2 (not 3)
                      [1, 1, 0]]]) # Query 3 can see Key 1, 2 (not 3)
    
    print(f"Applying Mask (Ignoring 3rd column):\n{mask}")
    
    masked_output, masked_weights = scaled_dot_product_attention(Q, K, V, mask=mask)
    
    print("\nMasked Weights (Column 3 should be 0):\n", np.round(masked_weights, 4))
    
    # Verification
    is_column_zero = np.allclose(masked_weights[0, :, 2], 0)
    print(f"\nVerification: Is the 3rd column effectively zero? {is_column_zero}")

if __name__ == "__main__":
    run_demo()
