import os
from collections import Counter, defaultdict
from typing import BinaryIO
import regex as re
import heapq
from multiprocessing import Pool
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class MaxHeapItem:
    def __init__(self, b1:bytes,b2:bytes, count:int):
        self.b1 = b1
        self.b2 = b2
        self.count = count

    def __lt__(self, other):
        if self.count != other.count:
            return self.count > other.count
        elif self.b1 != other.b1:
            return self.b1 > other.b1
        return self.b2 > other.b2
    def get_tuple(self) -> tuple[bytes, bytes]:
        return self.b1, self.b2

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
    pre_token_frequency_table = Counter()
    for result in results:
        pre_token_frequency_table.update(result)
    merges_pair = pair_table(pre_token_frequency_table = pre_token_frequency_table,  num_merges=vocab_size - len(vocab))
    for pair_x, pair_y in merges_pair:
        vocab[len(vocab)] = pair_x + pair_y
    return vocab, merges_pair

def pair_table(pre_token_frequency_table :Counter[tuple[bytes, ...]],
               num_merges) -> list[tuple[bytes , bytes]]:
    """
    预分词后，真正的bpe处理逻辑
    """
    merges_pair : list[tuple[bytes , bytes]] = []
    pair_counts, pair_to_words, pair_heap = init_pair_index(pre_token_frequency_table)
    for _ in range(num_merges):
        if not pair_counts:
            break
        # 选择出现频次最高的 pair 进行合并
        best_pair, best_pair_count = pop_heap_top_pairs(pair_heap)
        if best_pair is None or best_pair_count is None:
            break
        while pair_counts[best_pair] != best_pair_count:
            best_pair, best_pair_count = pop_heap_top_pairs(pair_heap)
            if best_pair is None or best_pair_count is None:
                break
        # print("111 ",len(merges_pair)," ",best_pair, best_pair_count , pair_heap[0].get_tuple(), pair_heap[0].count)
        merges_pair.append(best_pair)
        pre_token_frequency_table, pair_counts, pair_to_words,pair_heap = apply_merge(pre_token_frequency_table, pair_counts, pair_to_words, pair_heap, best_pair)
        # print("222 ",  pair_heap[0].get_tuple(), pair_heap[0].count)
    return merges_pair

def apply_merge(pre_token_frequency_table, pair_counts, pair_to_words, pair_heap, best_pair):
    # 要处理的词
    affected_words = list(pair_to_words.get(best_pair, {}).keys())
    if not affected_words:
        return pre_token_frequency_table, pair_counts, pair_to_words, pair_heap
    # {预分词：预分词数量}
    affected_words_count = {word: pre_token_frequency_table[word] for word in affected_words}
    changed_pairs = set()
    for old_word in affected_words:
        # 预分词数量
        pre_token_frequency_count = affected_words_count[old_word]
        # 返回一个词内部所有相邻 pair 的频次
        old_word_pair = pairs_in_word(old_word)
        # 中间数据 pair_counts pair_to_words 减去old_word 的所能构成的pair的数量
        for pair, counts in old_word_pair.items():
            changed_pairs.add(pair)
            pair_counts[pair] -= counts * pre_token_frequency_count
            if pair_counts[pair] <= 0:
                del pair_counts[pair]

            pair_to_words[pair][old_word] -= 1
            if pair_to_words[pair][old_word] <= 0:
                del pair_to_words[pair][old_word]
            if not pair_to_words[pair]:
                del pair_to_words[pair]
        # pre_token_frequency_table 关于预分词old_word 的数量删除
        pre_token_frequency_table[old_word] -= pre_token_frequency_count
        if pre_token_frequency_table[old_word] <= 0:
            del pre_token_frequency_table[old_word]

        new_word = merge_word(old_word, best_pair)

        pre_token_frequency_table[new_word] += pre_token_frequency_count

        new_pair_counts = pairs_in_word(new_word)
        for pair, counts in new_pair_counts.items():
            changed_pairs.add(pair)
            pair_counts[pair] += pre_token_frequency_count * counts
            pair_to_words[pair][new_word] += 1
    # 4. 由于heap 不能删除旧数据，每次更新又会涉及多个word
    # 为避免塞入脏数据，所以只能在所有预分词更新完成之后，再去更新heap中的pair_count
    for pair in changed_pairs:
        final_count = pair_counts.get(pair, 0)

        if final_count > 0:
            update_max_heap_pairs(
                pair_heap,
                pair[0],
                pair[1],
                final_count,
            )
    return pre_token_frequency_table, pair_counts, pair_to_words, pair_heap
    pass
def merge_word(word: tuple[bytes, ...], best_pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    """
    合并预分词
    """
    merged_token = best_pair[0] + best_pair[1]
    new_word = []
    i = 0
    n = len(word)
    while i < n:
        if i < n - 1 and (word[i], word[i + 1]) == best_pair:
            new_word.append(merged_token)
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    return tuple(new_word)
def init_pair_index(pre_token_frequency_table : Counter[tuple[bytes, ...]]):
    """
    构建：
    1. pair_counts: 每个 pair 在整个语料中出现的总频次
    2. pair_to_words: 每个 pair 出现在哪些词里
       这里 value 用 Counter 而不是 set，是为了处理：
       不同旧词 merge 后变成同一个 new_word 的碰撞情况
    3、pair_heap pair最大堆，先按照count排序，再按照字典序排序
    """
    pair_counts = Counter()
    pair_to_words = defaultdict(Counter)
    for word, count in pre_token_frequency_table.items():
        word_pair_counts = pairs_in_word(word)
        for pair, multiplicity in word_pair_counts.items():
            pair_counts[pair] += multiplicity * count
            pair_to_words[pair][word] += 1
    pair_heap = get_max_heap_pairs(pair_counts)
    return pair_counts, pair_to_words,pair_heap

def pairs_in_word(word:tuple[bytes, ...]) -> Counter[tuple[bytes, bytes]]:
    """
    返回一个词内部所有相邻 pair 的频次
        (b'a', b'b', b'a', b'b') -> {(b'a', b'b'): 2, (b'b', b'a'): 1}
    """
    # 不存在相邻对
    if len(word) < 2:
        return Counter()
    return Counter(zip(word, word[1:]))
def pair_list_in_word(word:tuple[bytes, ...]) -> list[tuple[bytes, bytes]]:
    if len(word) < 2:
        return []
    return list(zip(word, word[1:]))
def get_max_heap_pairs(pair_counts: Counter[tuple[bytes, bytes]]):
    """
    将 Counter 转换为最大堆，排序规则：
    1. 频次由高到低
    2. 频次相同时，按 tuple[bytes, bytes] 的字典序由大到小
    """
    heap = []

    for (b1, b2), count in pair_counts.items():
        heap_item = MaxHeapItem(b1, b2, count)
        heapq.heappush(heap, heap_item)

    return heap
def update_max_heap_pairs(heap, b1:bytes,b2:bytes, count:int):
    """
    新增pairs
    """
    heap_item = MaxHeapItem(b1, b2, count)
    heapq.heappush(heap, heap_item)
def pop_heap_top_pairs(heap: list[MaxHeapItem]) -> tuple[tuple[bytes, bytes], int]:
    """从堆中弹出，还原为 (pair, count) """
    if heap:
        heap_item = heapq.heappop(heap)
        return heap_item.get_tuple(), heap_item.count
    return None, None
def handel_chunk(
        input_path: str | os.PathLike,
        pattern: str,
        start: int,
        end: int,
        **kwargs,
) -> Counter[tuple[bytes, ...]]:
    """
    文件分块后，对每块进行字符统计
    """
    frequency_table = Counter()
    with (open(input_path, "rb") as f):
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        # 先剔除特殊字符
        remove_special_token_chunks = re.split(pattern, chunk)
        for one_chunk in remove_special_token_chunks:
            for match in re.finditer(PAT, one_chunk):
                bytes_str = match.group().encode('utf_8')
                pieces = tuple(bytes([x]) for x in bytes_str)
                frequency_table[pieces] +=1
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