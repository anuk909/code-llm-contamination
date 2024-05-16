from typing import List, Dict, Tuple, Union

ChunkResult = Dict[str, Union[str, int]]
DetailedResult = Dict[str, Union[str, int, List[ChunkResult]]]
FuzzyMatchResults = Dict[str, DetailedResult]
CorpusChunks = List[Tuple[int, str]]

DolosResult = Dict[str, Union[int, List[Dict[str, int]]]]
DolosResults = List[DolosResult]
