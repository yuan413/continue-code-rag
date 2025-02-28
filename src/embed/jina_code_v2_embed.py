import os
import lancedb
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import EmbeddingFunctionRegistry

os.environ['JINA_API_KEY'] = 'jina_*'

jina_embed = EmbeddingFunctionRegistry.get_instance().get("jina").create(name="jina-embeddings-v2-base-code")


class TextModel(LanceModel):
    text: str = jina_embed.SourceField()
    vector: Vector(jina_embed.ndims()) = jina_embed.VectorField()


data = [{"text": "hello world"},
        {"text": "goodbye world"}]

db = lancedb.connect("~/.lancedb-2")
tbl = db.create_table("test", schema=TextModel, mode="overwrite")

tbl.add(data)