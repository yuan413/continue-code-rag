import lancedb

db = lancedb.connect("~/.lancedb")

db.drop_table("test_merge_insert")

table = db.create_table(
    "test_merge_insert",
    [
        {"file_name": "db.py", "file_chunk": 1, "text": "db_chunk1", "vector": [0.01, 0.13]},
        {"file_name": "db.py", "file_chunk": 2, "text": "db_chunk2", "vector": [0.22, 0.33]},
        {"file_name": "db.py", "file_chunk": 3, "text": "db_chunk3", "vector": [0.67, 0.98]},
        {"file_name": "rpc.py", "file_chunk": 1, "text": "rpc_chunk1", "vector": [0.23, 0.45]},
        {"file_name": "rpc.py", "file_chunk": 2, "text": "rpc_chunk2", "vector": [0.43, 0.78]},
    ],
)

print(table.to_pandas())

new_data = [
        {"file_name": "db.py", "file_chunk": 1, "text": "db_chunk1_new", "vector": [0.11, 0.13]},
        {"file_name": "db.py", "file_chunk": 2, "text": "db_chunk2_new", "vector": [0.41, 0.03]},
        {"file_name": "rpc.py", "file_chunk": 1, "text": "rpc_chunk1_new", "vector": [0.09, 0.23]},
        {"file_name": "rpc.py", "file_chunk": 2, "text": "rpc_chunk2_new", "vector": [0.04, 0.12]},
        {"file_name": "rpc.py", "file_chunk": 3, "text": "rpc_chunk3_new", "vector": [0.21, 0.63]},
    ]

table.merge_insert(["file_name", "file_chunk"])\
    .when_not_matched_insert_all()\
    .when_matched_update_all()\
    .when_not_matched_by_source_delete()\
    .execute(new_data)

print(table.to_pandas())


