""" Contribution: https://github.com/lucidrains/DALLE2-pytorch """

import torch
import open_clip

_tokenizer = open_clip.get_tokenizer('hf-hub:wisdomik/QuiltNet-B-32')

class SimpleTokenizer:
    """
    A simple tokenizer for converting text into a sequence of tokens.

    This tokenizer is designed to process text inputs and convert them into tokens
    within a specified context length. It supports both single and multiple text inputs.

    Attributes:
        context_length (int): The fixed size to which the token sequences will be padded or truncated.
        truncate_text (bool): Flag indicating whether to truncate text that exceeds the context length.
    """

    def tokenize(self, texts, context_length: int = 256, truncate_text: bool = False) -> torch.Tensor:
        """
        Tokenizes the input text(s) and converts them into a tensor of token indices.

        Args:
            texts (str or List[str]): The input text(s) to be tokenized.
            context_length (int): The fixed size to which the token sequences will be padded or truncated.
            truncate_text (bool): If True, texts longer than `context_length` will be truncated.

        Returns:
            torch.Tensor: A tensor of token indices with shape (number of texts, context_length).

        Raises:
            RuntimeError: If any text is longer than `context_length` and `truncate_text` is False.
        """
        if isinstance(texts, str):
            texts = [texts]

        all_tokens = [_tokenizer.encode(text) for text in texts]
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                if truncate_text:
                    tokens = tokens[:context_length]
                else:
                    raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
            result[i, :len(tokens)] = torch.tensor(tokens)

        return result
    
tokenizer = SimpleTokenizer()
