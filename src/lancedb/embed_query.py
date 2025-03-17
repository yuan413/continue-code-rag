import lancedb
from lancedb.table import Table


class EmbeddingQuery:

    _table: Table

    def __init__(self, **kwargs):
        db = lancedb.connect("~/.lancedb")
        self._table = db.open_table("test_code_x_embed")

    def search(self, query_embedding: list):
        return self._table.search(query_embedding).limit(4).to_list()

