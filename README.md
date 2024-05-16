# Novelty Scoring Pipeline

This pipeline was built to find similar programs within the training data of large language models (LLMs). It works in two stages, using the RapidFuzz library and Dolos software to find programs that are similar both on a surface level and semantically.
This tool could be useful for verifying the novelty of exercises or searching test data in training.

## Table of Contents

- [Project Overview](#project-overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage Instructions](#usage-instructions)
  - [Setting Up the Corpus](#setting-up-the-corpus)
  - [Running Surface Level Similarity Score Pipeline](#running-surface-level-similarity-score-pipeline)
  - [Running Semantic Level Similarity Score Pipeline](#running-semantic-level-similarity-score-pipeline)
- [Configurable Parameters](#configurable-parameters)
- [Citation](#citation)

## Project Overview

This project extends an existing project to search for similar code in GitHub using shallow and semantic search methods with the Dolos tool.
The goal is to create novel Python exercises using GPT-4 and verify their novelty.

## Prerequisites

- Python 3.x
- Dependencies listed in `novelty_score/requirements.txt`
- Dolos tool (refer to [Dolos installation guide](https://dolos.ugent.be/docs/installation.html))

## Installation

1. **Clone the repository:**

   ```bash
   git clone git@github.com:anuk909/code-llm-contamination.git
   cd code-llm-contamination/
   ```

2. **Install Python dependencies:**

   ```bash
   pip install -r novelty_score/requirements.txt
   ```

3. **Install Dolos:**
   Follow the installation instructions provided in the [Dolos documentation](https://dolos.ugent.be/docs/installation.html).

## Usage Instructions

### Setting Up the Corpus

The `setup_corpus.py` script allows you to download and prepare the github python training dataset of the PILE.
It downloads parquet files from a specified URL and converts them to JSONL format for further processing.
It's possible to use other corpus in the same format.

1. **Run the setup script:**
   ```bash
   python novelty_score/setup_corpus_main.py
   ```

### Running Surface Level Similarity Score Pipeline

1. **Run the following commands to perform the surface level similarity check:**

   ```bash
   python novelty_score/fuzzy_match_main.py --input_path <input_path> --result_dir <result_dir> --max_corpus_files <max_corpus_files> --max_corpus_chunks <max_corpus_chunks> --detailed_results
   ```

2. **Example usage:**

   ```bash
   python novelty_score/fuzzy_match_main.py --input_path inputs/HumanEval.jsonl --result_dir results --max_corpus_files 1 --max_corpus_chunks 1 --detailed_results
   python novelty_score/fuzzy_match_main.py --input_path inputs/SingleHumanEval.jsonl --result_dir results --max_corpus_files 1 --max_corpus_chunks 40 --detailed_results
   ```

3. **Result format**
   The result file is jsonl format with each line correspond to other caononical_solution.
   without detailed_results it will look like that:

   ```json
   {
     "solution": " return n**2\n",
     "score": 81,
     "closest_solution": "))\n return n\n",
     "chunk_results": null
   }
   ```

   with detailed_results, chunk_results will be list of closest_solution in each chunk that passed FUZZ_THRESHOLD and will look like that:

   ```json
   {
     "solution": " return n**2\n",
     "score": 88,
     "closest_solution": " return 2**c\n",
     "chunk_results": [
       { "chunk_index": 0, "closest_solution": "))\n return n\n", "score": 81 },
       { "chunk_index": 2, "closest_solution": " return None\n", "score": 81 }
     ]
   }
   ```

   There are more examples in results/

### Running Semantic Level Similarity Score Pipeline

1. **Run the following command to perform the semantic similarity check:**

   ```bash
   python novelty_score/dolos_main.py --input_path <input_path> --result_dir <result_dir>
   ```

2. **Example usage:**

   ```bash
   python novelty_score/dolos_main.py --input_path inputs/FuzzyMatchHumanEval.jsonl --result_dir results
   python novelty_score/dolos_main.py --input_path inputs/FuzzyMatchHumanEval.jsonl --result_dir results
   ```

3. **Result format**
   The result file is jsonl format with each line correspond to other caononical_test.
   The output of it will look like that:

   ```json
   {"program_index": 14, "sorted_program_results": []}
   {"program_index": 15, "program_best_matches": [{"chunk_index": 4, "score": 59}, {"chunk_index": 6, "score": 10}]}
   ```

   Where program_index connected to the same line in the input file.

   There are more examples in results/

### Configurable Parameters

In novelty_score/utils/constants.py there are some configurable parameters that
it's possible to change according to your data.

- 'CHUNK_SIZE' = The maximal string length of each chunk from corpus data
- 'MAX_WORKERS' = Number of processes to use (default: 8).
- 'FUZZ_THRESHOLD' = The minimal relevant fuzz score to save.
- 'STRIDE_PERCENT' = The relative fuzz search jump, relevant only in slow_fuzzy_match

# Citation

#TODO - maybe add shmulik cohen's team

```
@misc{placeholder,
   author = {Martin Riddell and Ansong Ni and Arman Cohan},
   title = {Quantifying Contamination in Evaluating Code Generation Capabilities of Language Models},
   year={2024},
   publisher = {GitHub},
   journal = {GitHub repository},
   howpublished = {\url{https://github.com/yale-nlp/code-llm-contamination}}
}
```
