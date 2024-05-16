from io import StringIO
from tqdm import tqdm
from .fuzzy_matching import fast_fuzzy_match, fuzzy_match_shared_memory
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing.shared_memory import SharedMemory
from .constants import SHM_NAME, CHUNK_SIZE, MAX_WORKERS
from .logger import logger


def create_corpus_chunks(corpus_data, max_chunks, start_index=0):
    chunks, str_builder, i = [], StringIO(), 0
    while i < len(corpus_data) and (len(chunks) < max_chunks if max_chunks else True):
        while i < len(corpus_data) and str_builder.tell() < CHUNK_SIZE:
            str_builder.write(corpus_data[i])
            i += 1
        chunks.append((len(chunks) + start_index, str_builder.getvalue()))
    logger.info(f"Created {len(chunks)} corpus chunks")
    return chunks


def search_test_string_in_chunks(test_str, corpus_chunks, detailed_results):
    results = {
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
                results["chunk_results"].append(
                    {"chunk_index": idx, "closest_solution": match, "score": score}
                )
            if score > results["score"]:
                results.update({"closest_solution": match, "score": score})
                if score == 100:
                    break
    return results


def search_multiple_test_strings_in_chunks(
    test_strings, corpus_chunks, detailed_results
):
    results = {
        test_str: {
            "closest_solution": None,
            "score": 0,
            "chunk_results": [] if detailed_results else None,
        }
        for test_str in test_strings
    }
    for idx, chunk_str in tqdm(corpus_chunks, desc="Processing chunks"):
        shm = SharedMemory(
            name=SHM_NAME, create=True, size=len(chunk_str.encode("utf-8"))
        )
        shm.buf[: len(chunk_str.encode("utf-8"))] = chunk_str.encode("utf-8")
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
