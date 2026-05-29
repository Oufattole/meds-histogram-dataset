from importlib.metadata import PackageNotFoundError, version

__package_name__ = "meds_torchdata"

try:
    __version__ = version(__package_name__)
except PackageNotFoundError:  # pragma: no cover
    # package is not installed
    __version__ = "unknown"

from .config import MEDSTorchDataConfig
from .pytorch_dataset import MEDSPytorchDataset
from .types import MEDSTorchBatch

__all__ = ["MEDSPytorchDataset", "MEDSTorchBatch", "MEDSTorchDataConfig"]
