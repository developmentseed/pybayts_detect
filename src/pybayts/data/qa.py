from enum import Enum
import numpy
from rio_tiler.io import COGReader
import attr
from functools import partial


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


@attr.s
class LandsatCOGReader(COGReader):
    """Custom COG Reader."""

    def __attrs_post_init__(self):
        """Define _kwargs."""
        basename = os.path.basename(self.filepath)
        collection = basename.split("_")[5]
        QAValues = (
            Collection1_QAValues
            if collection == "01"
            else Collection2_QAValues
        )
        if "QA_PIXEL" in basename:
            self.resampling_method = "nearest"
            self.nodata = 1
            self.post_process = partial(pixel_to_qa, QAValues=QAValues)
        else:
            self.resampling_method = "bilinear"

        super().__attrs_post_init__()


def _capture_bits(arr, b1, b2, binary):
    width_int = int((b1 - b2 + 1) * "1", 2)
    out = ((arr >> b2) & width_int).astype("uint8")
    return out == 1 if binary else out == 3


def pixel_to_qa(
    data: numpy.ndarray, mask: numpy.ndarray, QAValues: Iterable
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """Convert DN to QA."""
    output_data = numpy.zeros(data.shape, dtype=numpy.uint16)
    for qa in QAValues:
        bits = qa.value["bits"]
        out = _capture_bits(data, bits[1], bits[0], qa.value["binary"])
        output_data[out] = qa.value["out"]

    return output_data, mask
