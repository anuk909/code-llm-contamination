from typing import List, Dict, Tuple, Union, Any

# Define a custom type alias for the results
ChunkResult = Dict[str, Union[str, int]]
DetailedResult = Dict[str, Union[str, int, List[ChunkResult]]]
FuzzyMatchResults = Dict[str, DetailedResult]

CorpusChunks = List[Tuple[int, str]]
