import numpy as np
from numpy.typing import NDArray

class FeatureExtractor:
    def __init__(self, sample_rate: int) -> None: ...
    def extract_features(
        self,
        audio: NDArray[np.float32],
        sample_rate: int,
    ) -> NDArray[np.float32]: ...
