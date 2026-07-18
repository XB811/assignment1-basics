import os
from typing import BinaryIO
import regex as re
from multiprocessing import Pool
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def train_bpe(
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str],
        num_processes : int = 1,
        **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes  , bytes]]]:
    vocab = _base_vocab(special_tokens)
    # 文本切块
    split_special_token = special_tokens[0].encode('utf_8')
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, split_special_token)
    pattern = "|".join(re.escape(tok) for tok in special_tokens)
    params = [(input_path, pattern, start, end) for start, end in zip(boundaries[:-1], boundaries[1:])]
    with Pool(num_processes) as pool:
        results = pool.starmap(handel_chunk, params)
    # 合并统计结果
    pre_token_frequency_table = {}
    for result in results:
        for key, count in result.items():
            pre_token_frequency_table[key] = pre_token_frequency_table.get(key, 0) + count
    print(pre_token_frequency_table.keys().__len__())
    merges_pair = pair_table(pre_token_frequency_table)
    for pair_x, pair_y in merges_pair:
        vocab[len(vocab)] = pair_x + pair_y
    return vocab, merges_pair

def pair_table(pre_token_frequency_table) -> list[tuple[bytes , bytes]]:
    """
    预分词后，真正的bpe处理逻辑
    """
    merges_pair = []

    return merges_pair

def handel_chunk(
        input_path: str | os.PathLike,
        pattern: str,
        start: int,
        end: int,
        **kwargs,
) -> dict[tuple[bytes, ...], int]:
    """
    文件分块后，对每块进行字符统计
    """
    frequency_table : dict[tuple[bytes, ...], int] = {}
    with (open(input_path, "rb") as f):
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        # 先剔除特殊字符
        remove_special_token_chunks = re.split(pattern, chunk)
        for one_chunk in remove_special_token_chunks:
            for match in re.finditer(PAT, one_chunk):
                bytes_str = match.group().encode('utf_8')
                pieces = tuple(bytes([x]) for x in bytes_str)
                frequency_table[pieces] = frequency_table.get(pieces, 0) + 1
    return frequency_table


def find_chunk_boundaries(
        file: BinaryIO,
        desired_num_chunks: int,
        split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def _base_vocab(
        special_tokens: list[str],
):
    """
    构建基础分词表 
    """
    vocab = dict[int, bytes]()
    for tokens in special_tokens:
        vocab[len(vocab)] =  tokens.encode('utf_8')
    for i in range(256):
        vocab[len(vocab)] = bytes([i])
    return vocab