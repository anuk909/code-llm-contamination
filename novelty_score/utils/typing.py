from typing import List, Dict, Tuple, Union, Any

# Define a custom type alias for the results
ChunkResult = Dict[str, Union[str, int]]
DetailedResult = Dict[str, Union[str, int, List[ChunkResult]]]
FuzzyMatchResults = Union[DetailedResult, Dict[str, DetailedResult]]

CorpusChunk = List[Tuple[int, str]]
CorpusChunkList = List[CorpusChunk]
