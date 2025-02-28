# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The LanceDB Authors


from typing import List, Union

import numpy as np
from lancedb.embeddings.base import TextEmbeddingFunction
from lancedb.embeddings.registry import register
from lancedb.embeddings.utils import weak_lru
from lancedb.util import attempt_import_or_raise


@register("voyageai_code")
class VoyageAICodeEmbeddingFunction(TextEmbeddingFunction):

    name: str = "voyage-code-3"
    device: str = "cpu"
    normalize: bool = True
    trust_remote_code: bool = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ndims = 2048

    @property
    def embedding_model(self):
        return self.get_embedding_model()

    def ndims(self):
        return self._ndims

    def generate_embeddings(
        self, texts: Union[List[str], np.ndarray]
    ) -> List[np.array]:
        return self.generate_text_embeddings(texts)

    def generate_text_embeddings(self, text: str, **kwargs) -> np.ndarray:
        rs = self.embedding_model.embed(
            list(text),
            model=self.name,
            input_type="document",
            truncation=True,
            output_dimension=self._ndims
        )
        return rs.embeddings

    @weak_lru(maxsize=1)
    def get_embedding_model(self):
        voyageai = attempt_import_or_raise("voyageai")
        return voyageai.Client()
