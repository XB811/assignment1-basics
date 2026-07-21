from __future__ import annotations

import argparse
import cProfile
import os
import pickle
import pstats
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import BinaryIO, TextIO
import regex as re
import heapq
from multiprocessing import Pool

from tqdm import tqdm

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
BYTE_SYMBOLS: tuple[bytes, ...] = tuple(bytes((i,)) for i in range(256))
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
        worker_profile_dir: str | os.PathLike | None = None,
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
        if worker_profile_dir is None:
            results = pool.starmap(handel_chunk, params)
        else:
            resolved_profile_dir = Path(worker_profile_dir).resolve()
            resolved_profile_dir.mkdir(parents=True, exist_ok=True)
            profiled_params = [
                (*param, str(resolved_profile_dir))
                for param in params
            ]
            results = pool.starmap(profiled_handel_chunk, profiled_params)
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
        pre_token_counts: Counter[str] = Counter()

        special_pattern = re.compile(pattern)
        pre_token_pattern = re.compile(PAT)

        for one_chunk in special_pattern.split(chunk):
            pre_token_counts.update(
                match.group()
                for match in pre_token_pattern.finditer(one_chunk)
            )

        frequency_table = Counter({
            tuple(BYTE_SYMBOLS[x] for x in token.encode("utf-8")): count
            for token, count in pre_token_counts.items()
        })
    return frequency_table


def profiled_handel_chunk(
        input_path: str | os.PathLike,
        pattern: str,
        start: int,
        end: int,
        profile_dir: str | os.PathLike,
) -> Counter[tuple[bytes, ...]]:
    """Profile one worker task and retain its result unchanged."""
    profiler = cProfile.Profile()
    profile_path = (
        Path(profile_dir)
        / f"worker_{os.getpid()}_{start}_{end}.prof"
    )

    try:
        return profiler.runcall(handel_chunk, input_path, pattern, start, end)
    finally:
        profiler.dump_stats(str(profile_path))


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



def save_vocab(vocab: dict[int, bytes], output_path: str | os.PathLike):
    with open(output_path, "wb") as f:
        pickle.dump(vocab, f)

def save_merges(merges: list[tuple[bytes,bytes]], output_path: str | os.PathLike):
    with open(output_path, "wb") as f:
        pickle.dump(merges, f)

def main(worker_profile_dir: str | os.PathLike | None = None):
    input_path = "data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]
    num_processes = 28

    start = datetime.now()
    vocab, merges = train_bpe(
        input_path,
        vocab_size,
        special_tokens,
        num_processes,
        worker_profile_dir=worker_profile_dir,
    )
    end = datetime.now()
    tqdm.write(f"time: {end - start}")
    tqdm.write("Saving outputs")
    save_vocab(vocab, "data/tinystories_vocab.pkl")
    save_merges(merges, "data/tinystories_merges.pkl")


def merge_worker_profiles(
        profile_dir: str | os.PathLike,
        output_path: str | os.PathLike,
        stream: TextIO | None = None,
) -> int:
    """Merge per-task profiles and return the number of merged files."""
    profile_paths = sorted(Path(profile_dir).glob("*.prof"))
    if not profile_paths:
        return 0

    if stream is not None:
        stream.write("Per-task worker profile totals:\n")
        for profile_path in profile_paths:
            task_stats = pstats.Stats(str(profile_path), stream=stream)
            stream.write(f"{profile_path.name}: {task_stats.total_tt:.6f} seconds\n")
        stream.write(
            "\nMerged times below are sums across workers, not wall-clock time.\n\n"
        )

    stats = pstats.Stats(str(profile_paths[0]), stream=stream)
    for profile_path in profile_paths[1:]:
        stats.add(str(profile_path))

    stats.strip_dirs().sort_stats("cumulative")
    stats.dump_stats(str(output_path))
    if stream is not None:
        stats.print_stats()
    return len(profile_paths)


def run_with_profile(profile_dir: str | os.PathLike = "profiles"):
    """Run training and retain both machine-readable and text profile reports."""
    output_dir = Path(profile_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    profile_path = output_dir / f"train_bpe_{timestamp}.prof"
    log_path = output_dir / f"train_bpe_{timestamp}.log"
    worker_profile_dir = output_dir / f"workers_{timestamp}"
    worker_profile_path = output_dir / f"train_bpe_workers_{timestamp}.prof"
    worker_log_path = output_dir / f"train_bpe_workers_{timestamp}.log"

    profiler = cProfile.Profile()
    started_at = datetime.now()
    error: BaseException | None = None

    try:
        profiler.runcall(main, worker_profile_dir)
    except BaseException as exc:
        error = exc
        raise
    finally:
        finished_at = datetime.now()
        profiler.dump_stats(profile_path)

        worker_profile_count = 0
        worker_profile_error: Exception | None = None
        try:
            with worker_log_path.open("w", encoding="utf-8") as worker_log_file:
                worker_profile_count = merge_worker_profiles(
                    worker_profile_dir,
                    worker_profile_path,
                    stream=worker_log_file,
                )
        except Exception as exc:
            worker_profile_error = exc

        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write(f"Started: {started_at.isoformat()}\n")
            log_file.write(f"Finished: {finished_at.isoformat()}\n")
            log_file.write(f"Elapsed: {finished_at - started_at}\n")
            if error is None:
                log_file.write("Status: completed\n")
            else:
                log_file.write(f"Status: failed ({type(error).__name__}: {error})\n")
            log_file.write(f"Profile data: {profile_path.resolve()}\n")
            log_file.write(
                f"Worker profiles: {worker_profile_dir.resolve()} "
                f"({worker_profile_count} files)\n"
            )
            if worker_profile_count:
                log_file.write(
                    f"Merged worker profile: {worker_profile_path.resolve()}\n"
                )
                log_file.write(
                    f"Merged worker log: {worker_log_path.resolve()}\n"
                )
            if worker_profile_error is not None:
                log_file.write(
                    "Worker profile merge failed: "
                    f"{type(worker_profile_error).__name__}: {worker_profile_error}\n"
                )
            log_file.write("\n")

            pstats.Stats(profiler, stream=log_file).strip_dirs().sort_stats("cumulative").print_stats()

        tqdm.write(f"Profile data saved to {profile_path.resolve()}")
        tqdm.write(f"Profile log saved to {log_path.resolve()}")
        if worker_profile_count:
            tqdm.write(
                f"Merged worker profile saved to {worker_profile_path.resolve()}"
            )
            tqdm.write(
                f"Merged worker profile log saved to {worker_log_path.resolve()}"
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Train a byte-pair encoding tokenizer.")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="profile the training run and save reports under profiles/",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.profile:
        run_with_profile()
    else:
        main()
