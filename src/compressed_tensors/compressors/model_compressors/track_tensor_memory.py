from typing import Callable, Any, Type, List, Set
from functools import partial

import gc
import torch
import weakref


class TrackTensorAllocations:
    total_tensor_memory: int
    memory_timeline: List[int]
    
    _tracked_tensors: Set[int]
    _original_init_fn: Callable[[Any], None]

    def __init__(self):
        self.total_tensor_memory = 0
        self.memory_timeline = []

        self._tracked_tensors = set()
        self._original_init_fn = torch.Tensor.__init__

    def __enter__(self):
        def wrapped_init(instance, *args, **kwargs):
            if isinstance(instance, torch.Tensor):
                self._original_init_fn(instance)
                self.track_tensor(instance)
            else:
                # parameters, ect.
                type(instance).__init__(instance, *args, **kwargs)
        
        torch.Tensor.__init__ = wrapped_init

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.Tensor.__init__ = self._original_init_fn
        self._active = False
        gc.collect()

    def track_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor_hash = hash(tensor)
        tensor_memory = tensor.numel() * tensor.element_size()

        # warn when init is called twice
        if tensor_hash in self._tracked_tensors:
            print("double init")
            return

        # add memory
        self.total_tensor_memory += tensor_memory
        self._add_to_timeline()
        self._tracked_tensors.add(tensor_hash)

        # register hook to subtract memory
        weakref.finalize(tensor, partial(self._on_tensor_deallocated, tensor_memory, tensor_hash))

    def _on_tensor_deallocated(self, tensor_memory, tensor_hash):
        self.total_tensor_memory -= tensor_memory
        self._add_to_timeline()
        self._tracked_tensors.remove(tensor_hash)
    
    @property
    def total_tensor_memory_mib(self):
        return self.total_tensor_memory / (1024 * 1024)
    
    def _add_to_timeline(self):
        self.memory_timeline.append(self.total_tensor_memory)
