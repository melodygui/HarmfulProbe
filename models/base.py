# abstract base class for different probe architectures 

from abc import ABC, abstractmethod
from pathlib import Path

import torch 

class BaseProbe(ABC):
    @abstractmethod
    def fit(self, X: list[torch.Tensor] | torch.Tensor, y: torch.Tensor, show_progress: bool = True):
        """
            @param X: input, either a list of tensor or a single tensor
            @pagram y : output, labels
            @pagram show_progress: optional argument to show progress during fitting
        """
        raise NotImplementedError
    
    @abstractmethod
    def predict(self, X: list[torch.Tensor] | torch.Tensor):
        raise NotImplementedError
    
    @abstractmethod
    def save(self, path: str | Path):
        raise NotImplementedError
   
    @abstractmethod
    def get_concept_vectors(self):
        """
            retreive latent representations of harmfulneess concept 
        """
        raise NotImplementedError
