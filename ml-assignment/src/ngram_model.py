import random
import re
from collections import defaultdict, Counter

class TrigramModel:
    def __init__(self):
        """
        Initializes the TrigramModel.
        """
        # TODO: Initialize any data structures you need to store the n-gram counts.
        # We use a nested dictionary structure:
        # model[word_1][word_2] = Counter({word_3: count, word_4: count})
        # This allows O(1) lookups for the next possible words.
        self.model = defaultdict(lambda: defaultdict(Counter))
    
    def _tokenize(self, text):
        """
        Helper method: Cleans text, splits into sentences, and adds padding.
        """
        # 1. Lowercase
        text = text.lower()
        
        # 2. Tokenize using Regex
        # Matches words (\w+) OR punctuation ([.!?])
        # We treat punctuation as separate tokens to help detect sentence boundaries.
        raw_tokens = re.findall(r"\w+|[.!?]", text)
        
        # 3. Add Padding (<START>, <END>)
        # Logic: Every sentence starts with 2 START tokens. 
        # Punctuation marks end the current sentence.
        tokens = ['<START>', '<START>']
        
        for token in raw_tokens:
            tokens.append(token)
            if token in ['.', '!', '?']:
                # End of sentence detected
                tokens.append('<END>')
                # Start of new sentence
                tokens.append('<START>')
                tokens.append('<START>')
        
        # Clean up trailing start tokens if the text ended with punctuation
        while tokens[-1] == '<START>':
            tokens.pop()
            
        return tokens

    def fit(self, text):
        """
        Trains the trigram model on the given text.

        Args:
            text (str): The text to train the model on.
        """
        # TODO: Implement the training logic.
        # This will involve:
        # 1. Cleaning the text (e.g., converting to lowercase, removing punctuation).
        # 2. Tokenizing the text into words.
        # 3. Padding the text with start and end tokens.
        # 4. Counting the trigrams.
        
        print("Preprocessing and Tokenizing...")
        tokens = self._tokenize(text)
        
        # To make the model robust, we convert words that appear only once to <UNK>.
        # This helps the model handle words it hasn't seen before during generation.
        word_counts = Counter(tokens)
        # Threshold: Words must appear at least 2 times, or they are <UNK>
        # We preserve <START> and <END> regardless of count
        tokens = [
            t if (word_counts[t] >= 2 or t in ('<START>', '<END>')) else '<UNK>' 
            for t in tokens
        ]
        
        print(f"Training on {len(tokens)} tokens...")
        
        # Sliding window of size 3
        # We look at tokens [i] and [i+1] to predict [i+2]
        for i in range(len(tokens) - 2):
            w1 = tokens[i]
            w2 = tokens[i+1]
            w3 = tokens[i+2]
            
            # We don't want to predict what comes after <END>, 
            # as we reset to <START> immediately after.
            if w1 == '<END>' or w2 == '<END>':
                continue
                
            self.model[w1][w2][w3] += 1
            
        print("Training complete.")

    def generate(self, max_length=50):
        """
        Generates new text using the trained trigram model.

        Args:
            max_length (int): The maximum length of the generated text.

        Returns:
            str: The generated text.
        """
        # TODO: Implement the generation logic.
        # This will involve:
        # 1. Starting with the start tokens.
        # 2. Probabilistically choosing the next word based on the current context.
        # 3. Repeating until the end token is generated or the maximum length is reached.
        
        
        all_contexts = []
        for u in self.model:
            for v in self.model[u]:
                all_contexts.append((u, v))

        if not all_contexts:
            return "Error: Model has not been trained."

        # Pick a random bigram to start
        w1, w2 = random.choice(all_contexts)
        
        # Initialize generated_words with our seed words
        # We filter out <START> so it doesn't appear in the final output string
        generated_words = []
        if w1 != '<START>': generated_words.append(w1)
        if w2 != '<START>': generated_words.append(w2)

        # We subtract the length of the seed from max_length
        for _ in range(max_length - len(generated_words)):
            # 2. Fetch possible next words
            possible_next_words = self.model[w1][w2]

            # Edge case: Dead end
            if not possible_next_words:
                break

            # Extract words and counts
            words = list(possible_next_words.keys())
            counts = list(possible_next_words.values())

            # Calculate Probabilities
            total_count = sum(counts)
            probs = [c / total_count for c in counts]

            # 3. Probabilistically choose the next word
            next_word = random.choices(words, weights=probs, k=1)[0]

            # If we hit an end token, stop
            if next_word == '<END>':
                break

            generated_words.append(next_word)

            # Shift the context window
            w1 = w2
            w2 = next_word

        return " ".join(generated_words)
