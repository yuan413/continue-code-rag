from voyageai_code_embed import VoyageAICodeEmbeddingFunction

import lancedb
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import EmbeddingFunctionRegistry

voyageai = EmbeddingFunctionRegistry.get_instance().get("voyageai_code").create()


class TextModel(LanceModel):
    filename: str
    text: str = voyageai.SourceField()
    vector: Vector(voyageai.ndims()) = voyageai.VectorField()


data = [
    {"text": "print('hello world!')", "filename": "hello.py"},
    {"text": "print('goodbye world!')", "filename": "goodbye.py"}
]

db = lancedb.connect("~/.lancedb")
tbl = db.create_table("test1", schema=TextModel, mode="overwrite")

tbl.add(data)

query = "greetings"
actual = tbl.search(query).limit(1).to_pydantic(TextModel)[0]
print(actual.text)
