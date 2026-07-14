import os
import regex as re
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def train_bpe(
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str],
        **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab = _base_vocab(special_tokens)
    print(len(vocab))
    print(vocab)
    return tuple(vocab, None)

def _base_vocab(
        special_tokens: list[str],
):
    """
    构建基础分词表
    """
    vocab = dict[int, bytes]()
    for tokens in special_tokens:
        _add_char(vocab, tokens)
    for i in range(256):
        _add_char(vocab, chr(i).encode('utf_8'))
    return vocab
def _add_char(
        vocab: dict[int, bytes],
        char: bytes
):
    vocab[len(vocab)] = char