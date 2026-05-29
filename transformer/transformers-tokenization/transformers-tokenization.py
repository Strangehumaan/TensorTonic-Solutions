import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """

        vocab = set(
            word
            for text in texts
            for word in text.lower().split()
        )

        vocab_with_tag = [
            self.pad_token,
            self.unk_token,
            self.bos_token,
            self.eos_token
        ] + sorted(vocab)

        self.word_to_id = {
            word: idx
            for idx, word in enumerate(vocab_with_tag)
        }

        self.id_to_word = {
            idx: word
            for idx, word in enumerate(vocab_with_tag)
        }

        self.vocab_size = len(vocab_with_tag)
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """

        words = text.lower().split()

        idxs = [
            self.word_to_id.get(
                word,
                self.word_to_id[self.unk_token]
            )
            for word in words
        ]

        return idxs
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """

        words = [
            self.id_to_word.get(id, self.unk_token)
            for id in ids
        ]

        return " ".join(words)