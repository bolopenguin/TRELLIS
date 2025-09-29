from abc import abstractmethod
from contextlib import contextmanager
import torch


class MemoryController:
    """
    Base class for memory management during training.
    """

    _last_input_size = None
    _last_mem_ratio = []

    @contextmanager
    def record(self):
        pass

    def update_run_states(self, input_size=None, mem_ratio=None):
        if self._last_input_size is None:
            self._last_input_size = input_size
        elif self._last_input_size != input_size:
            raise ValueError(
                f"Input size should not change for different ElasticModules."
            )
        self._last_mem_ratio.append(mem_ratio)

    @abstractmethod
    def get_mem_ratio(self, input_size):
        pass

    @abstractmethod
    def state_dict(self):
        pass

    @abstractmethod
    def log(self):
        pass


class ElasticModuleMixin:
    """
    Mixin for training with elastic memory management.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._memory_controller: MemoryController = None

    @abstractmethod
    def _get_input_size(self, *args, **kwargs) -> int:
        """
        Get the size of the input data.

        Returns:
            int: The size of the input data.
        """
        pass

    @abstractmethod
    @contextmanager
    def with_mem_ratio(self, mem_ratio=1.0) -> float:
        """
        Context manager for training with a reduced memory ratio compared to the full memory usage.

        Returns:
            float: The exact memory ratio used during the forward pass.
        """
        pass

    def register_memory_controller(self, memory_controller: MemoryController):
        self._memory_controller = memory_controller

    def forward(self, *args, **kwargs):
        if (
            self._memory_controller is None
            or not torch.is_grad_enabled()
            or not self.training
        ):
            ret = super().forward(*args, **kwargs)
        else:
            input_size = self._get_input_size(*args, **kwargs)
            mem_ratio = self._memory_controller.get_mem_ratio(input_size)
            with self.with_mem_ratio(mem_ratio) as exact_mem_ratio:
                ret = super().forward(*args, **kwargs)
            self._memory_controller.update_run_states(input_size, exact_mem_ratio)
        return ret
