import lancedb

db = lancedb.connect("~/.lancedb/test")

table = db.open_table("test_merge_insert")

print(table.to_pandas())