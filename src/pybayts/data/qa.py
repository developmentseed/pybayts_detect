from enum import Enum
import numpy
from typing import Iterable


class Collection2_QAValues(Enum):
    """Collection 2 QA band info."""

    DesignatedFill = {"out": 1, "bits": (0, 0), "binary": True}
    DilatedCloud = {"out": 8, "bits": (1, 1), "binary": True}
    Cirrus = {"out": 7, "bits": (2, 2), "binary": True}
    Cloud = {"out": 4, "bits": (3, 3), "binary": True}
    CloudShadow = {"out": 5, "bits": (4, 4), "binary": True}
    SnowIce = {"out": 6, "bits": (5, 5), "binary": True}
    Clear = {"out": 0, "bits": (6, 6), "binary": True}
    Water = {"out": 9, "bits": (7, 7), "binary": True}


def _capture_bits(arr, b1, b2, binary):
    width_int = int((b1 - b2 + 1) * "1", 2)
    out = ((arr >> b2) & width_int).astype("uint8")
    return out == 1 if binary else out == 3


def pixel_to_qa(data: numpy.ndarray, QAValues: Iterable) -> numpy.ndarray:
    """Convert DN to QA."""
    output_data = numpy.zeros(data.shape, dtype=numpy.uint16)
    for qa in QAValues:
        bits = qa.value["bits"]
        out = _capture_bits(data, bits[1], bits[0], qa.value["binary"])
        output_data[out] = qa.value["out"]

    return output_data
