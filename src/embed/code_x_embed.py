from typing import List, Union, Any

import numpy as np
from lancedb.embeddings.base import TextEmbeddingFunction
from lancedb.embeddings.registry import register
from lancedb.embeddings.utils import weak_lru
from lancedb.util import attempt_import_or_raise
import torch
from functools import cached_property
from src.util.custom_thread_local import thread_local
from pydantic import PrivateAttr


@register("code-x-embed")
class CodeXEmbeddingFunction(TextEmbeddingFunction):

    name: str = "/Users/01428674/Ai/model-hub/SFR-Embedding-Code-2B_R"
    device: str = "cpu"
    normalize: bool = True
    trust_remote_code: bool = True
    max_length: int = 32768

    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        self._ndims = 2304

    @property
    def embedding_model(self):
        """
        Get the sentence-transformers embedding model specified by the
        name, device, and trust_remote_code. This is cached so that the
        model is only loaded once per process.
        """
        return self.get_embedding_model()

    def ndims(self):
        return self._ndims

    def generate_embeddings(self, texts: Union[List[str], np.ndarray]) -> List[np.array]:
        return self.generate_text_embeddings(texts)

    def generate_text_embeddings(self, text: str, **kwargs) -> np.ndarray:
        rs = self.embedding_model.encode_corpus(
            list(text),
            max_length=self.max_length
        ).tolist()
        return rs

    @weak_lru(maxsize=1)
    def get_embedding_model(self):
        """
        Get the sentence-transformers embedding model specified by the
        name, device, and trust_remote_code. This is cached so that the
        model is only loaded once per process.

        TODO: use lru_cache instead with a reasonable/configurable maxsize
        """
        transformers = attempt_import_or_raise("transformers")
        model = transformers.AutoModel.from_pretrained(self.name, trust_remote_code=self.trust_remote_code,
                                                             low_cpu_mem_usage=False).to(self.device)
        return model
