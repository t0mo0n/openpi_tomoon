import json
import pathlib

import numpy as np
import numpydantic
import pydantic

try:
    import cupy
    CUPY_AVAILABLE = cupy.is_available()
except ImportError:
    cupy = None
    CUPY_AVAILABLE = False

@pydantic.dataclasses.dataclass
class NormStats:
    mean: numpydantic.NDArray
    std: numpydantic.NDArray
    q01: numpydantic.NDArray | None = None  # 1st quantile
    q99: numpydantic.NDArray | None = None  # 99th quantile


class RunningStats:
    """Compute running statistics of a batch of vectors."""

    def __init__(self, use_gpu: bool = False):
        self._count = 0
        self._mean = None
        self._mean_of_squares = None
        self._min = None
        self._max = None
        self._histograms = None
        self._bin_edges = None
        self._num_quantile_bins = 5000  # for computing quantiles on the fly

        # 选择计算后端 (cupy 或 numpy)
        if use_gpu and CUPY_AVAILABLE:
            self._backend = cupy
            self._device = "gpu"
            print("RunningStats: Using CuPy for GPU acceleration.")
        else:
            self._backend = np
            self._device = "cpu"
            if use_gpu:
                print("RunningStats: CuPy not available or no GPU found. Falling back to NumPy on CPU.")


    def update(self, batch: np.ndarray) -> None:
        """
        Update the running statistics with a batch of vectors.

        Args:
            vectors (np.ndarray): An array where all dimensions except the last are batch dimensions.
        """
        # 将输入数据移动到选择的设备上
        batch = self._backend.asarray(batch)

        batch = batch.reshape(-1, batch.shape[-1])
        num_elements, vector_length = batch.shape
        if self._count == 0:
            self._mean = self._backend.mean(batch, axis=0)
            self._mean_of_squares = self._backend.mean(batch**2, axis=0)
            self._min = self._backend.min(batch, axis=0)
            self._max = self._backend.max(batch, axis=0)
            self._histograms = [self._backend.zeros(self._num_quantile_bins) for _ in range(vector_length)]
            # np.linspace 在 cupy 中也可用
            self._bin_edges = [
                self._backend.linspace(self._min[i] - 1e-10, self._max[i] + 1e-10, self._num_quantile_bins + 1)
                for i in range(vector_length)
            ]
        else:
            if vector_length != self._mean.size:
                raise ValueError("The length of new vectors does not match the initialized vector length.")
            new_max = self._backend.max(batch, axis=0)
            new_min = self._backend.min(batch, axis=0)
            max_changed = self._backend.any(new_max > self._max)
            min_changed = self._backend.any(new_min < self._min)
            self._max = self._backend.maximum(self._max, new_max)
            self._min = self._backend.minimum(self._min, new_min)

            if max_changed or min_changed:
                self._adjust_histograms()

        self._count += num_elements

        batch_mean = self._backend.mean(batch, axis=0)
        batch_mean_of_squares = self._backend.mean(batch**2, axis=0)

        # Update running mean and mean of squares.
        self._mean += (batch_mean - self._mean) * (num_elements / self._count)
        self._mean_of_squares += (batch_mean_of_squares - self._mean_of_squares) * (num_elements / self._count)

        self._update_histograms(batch)

    def get_statistics(self) -> NormStats:
        """
        Compute and return the statistics of the vectors processed so far.

        Returns:
            dict: A dictionary containing the computed statistics.
        """
        if self._count < 2:
            raise ValueError("Cannot compute statistics for less than 2 vectors.")

        variance = self._mean_of_squares - self._mean**2
        stddev = self._backend.sqrt(self._backend.maximum(0, variance))
        q01, q99 = self._compute_quantiles([0.01, 0.99])

        # 将结果从 GPU 转回 CPU (如果需要)
        if self._device == "gpu":
            mean_np = self._mean.get()
            stddev_np = stddev.get()
            q01_np = q01.get()
            q99_np = q99.get()
        else:
            mean_np = self._mean
            stddev_np = stddev
            q01_np = q01
            q99_np = q99

        return NormStats(mean=mean_np, std=stddev_np, q01=q01_np, q99=q99_np)

    def _adjust_histograms(self):
        """Adjust histograms when min or max changes."""
        for i in range(len(self._histograms)):
            old_edges = self._bin_edges[i]
            new_edges = self._backend.linspace(self._min[i], self._max[i], self._num_quantile_bins + 1)

            # Redistribute the existing histogram counts to the new bins
            new_hist, _ = self._backend.histogram(old_edges[:-1], bins=new_edges, weights=self._histograms[i])

            self._histograms[i] = new_hist
            self._bin_edges[i] = new_edges

    def _update_histograms(self, batch) -> None:
        """Update histograms with new vectors."""
        for i in range(batch.shape[1]):
            hist, _ = self._backend.histogram(batch[:, i], bins=self._bin_edges[i])
            self._histograms[i] += hist

    def _compute_quantiles(self, quantiles):
        """Compute quantiles based on histograms."""
        results = []
        for q in quantiles:
            target_count = q * self._count
            q_values = []
            for hist, edges in zip(self._histograms, self._bin_edges, strict=True):
                cumsum = self._backend.cumsum(hist)
                idx = self._backend.searchsorted(cumsum, target_count)
                q_values.append(edges[idx])
            results.append(self._backend.array(q_values))
        return results


class _NormStatsDict(pydantic.BaseModel):
    norm_stats: dict[str, NormStats]


def serialize_json(norm_stats: dict[str, NormStats]) -> str:
    """Serialize the running statistics to a JSON string."""
    return _NormStatsDict(norm_stats=norm_stats).model_dump_json(indent=2)


def deserialize_json(data: str) -> dict[str, NormStats]:
    """Deserialize the running statistics from a JSON string."""
    return _NormStatsDict(**json.loads(data)).norm_stats


def save(directory: pathlib.Path | str, norm_stats: dict[str, NormStats]) -> None:
    """Save the normalization stats to a directory."""
    path = pathlib.Path(directory) / "norm_stats.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialize_json(norm_stats))


def load(directory: pathlib.Path | str) -> dict[str, NormStats]:
    """Load the normalization stats from a directory."""
    path = pathlib.Path(directory) / "norm_stats.json"
    if not path.exists():
        raise FileNotFoundError(f"Norm stats file not found at: {path}")
    return deserialize_json(path.read_text())
