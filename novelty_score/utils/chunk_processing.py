from typing import List, Tuple, Dict, Any
from io import StringIO
from tqdm import tqdm
from .fuzzy_matching import fast_fuzzy_match, fuzzy_match_shared_memory
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing.shared_memory import SharedMemory
from .constants import SHM_NAME, CHUNK_SIZE, MAX_WORKERS
from .logger import logger
from .typing import CorpusChunks


def create_corpus_chunks(
    corpus_data: List[str], max_corpus_chunks: int, start_index: int = 0
) -> CorpusChunks:
    chunks: CorpusChunks = []
    i: int = 0

    while i < len(corpus_data) and (
        len(chunks) < max_corpus_chunks if max_corpus_chunks else True
    ):
        with StringIO() as str_builder:
            while i < len(corpus_data) and str_builder.tell() < CHUNK_SIZE:
                str_builder.write(corpus_data[i])
                i += 1
            chunks.append((len(chunks) + start_index, str_builder.getvalue()))

    logger.info(f"Created {len(chunks)} corpus chunks")
    return chunks


def search_test_string_in_chunks(
    test_str: str, corpus_chunks: CorpusChunks, detailed_results: bool
) -> Dict[str, Any]:
    results: Dict[str, Any] = {
        "closest_solution": None,
        "score": 0,
        "chunk_results": [] if detailed_results else None,
    }
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(fast_fuzzy_match, test_str, chunk_str): idx
            for idx, chunk_str in corpus_chunks
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing chunks"
        ):
            match, score = future.result()
            idx = futures[future]
            if detailed_results:
                if match:
                    results["chunk_results"].append(
                        {"chunk_index": idx, "closest_solution": match, "score": score}
                    )
            if score > results["score"]:
                results.update({"closest_solution": match, "score": score})
                if score == 100:
                    break
    return results


def search_multiple_test_strings_in_chunks(
    test_strings: List[str],
    corpus_chunks: CorpusChunks,
    detailed_results: bool,
) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {
        test_str: {
            "closest_solution": None,
            "score": 0,
            "chunk_results": [] if detailed_results else None,
        }
        for test_str in test_strings
    }
    for idx, chunk_str in tqdm(corpus_chunks, desc="Processing chunks"):
        chunk_encoded: bytes = chunk_str.encode("utf-8")
        shm = SharedMemory(name=SHM_NAME, create=True, size=len(chunk_encoded))
        shm.buf[: len(chunk_encoded)] = chunk_encoded
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(fuzzy_match_shared_memory, test_str): test_str
                for test_str in test_strings
            }
            for future in as_completed(futures):
                match, score = future.result()
                test_str = futures[future]
                if detailed_results and match:
                    results[test_str]["chunk_results"].append(
                        {"chunk_index": idx, "closest_solution": match, "score": score}
                    )
                if score > results[test_str]["score"]:
                    results[test_str].update(
                        {"closest_solution": match, "score": score}
                    )
        shm.close()
        shm.unlink()
    return results
