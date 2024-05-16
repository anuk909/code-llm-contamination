from typing import Tuple, Optional
from rapidfuzz import fuzz
from multiprocessing.shared_memory import SharedMemory
from .constants import SHM_NAME, FUZZ_THRESHOLD, STRIDE_PERCENT


def slow_fuzzy_match(test_str: str, chunk: str) -> Tuple[Optional[str], int]:
    best_score = 0
    best_match: Optional[str] = None
    stride = max(int(len(test_str) * STRIDE_PERCENT), 1)
    for i in range(0, len(chunk) - len(test_str), stride):
        score = fuzz.ratio(chunk[i : i + len(test_str)], test_str)
        if score > best_score:
            best_score = round(score)
            best_match = chunk[i : i + len(test_str)]
    return (best_match, best_score) if best_score >= FUZZ_THRESHOLD else (None, 0)


def fast_fuzzy_match(test_str: str, chunk: str) -> Tuple[Optional[str], int]:
    score_alignment = fuzz.partial_ratio_alignment(test_str, chunk)
    if score_alignment and score_alignment.score > FUZZ_THRESHOLD:
        return (
            chunk[score_alignment.dest_start : score_alignment.dest_end],
            round(score_alignment.score),
        )
    else:
        return (None, 0)


def fuzzy_match_shared_memory(test_str: str) -> Tuple[Optional[str], int]:
    existing_shm = SharedMemory(name=SHM_NAME)
    chunk = existing_shm.buf[:].tobytes().decode("utf-8")
    existing_shm.close()
    return fast_fuzzy_match(test_str, chunk)
