import lancedb
import torch
import os
from lancedb.pydantic import LanceModel, Vector
from transformers import AutoModel
from src.spliter.sweep_ai_code_spliter import chunk_code


class TextModel(LanceModel):
    file_name: str
    text: str
    chunk_no: int
    vector: Vector(2304)

db = lancedb.connect("~/.lancedb")

db.drop_table("test_code_x_embed")
# table = db.open_table("test_code_x_embed")
table = db.create_table("test_code_x_embed", schema=TextModel)

device = torch.device("cpu")
max_length = 32768
model = AutoModel.from_pretrained('/Users/01428674/Ai/model-hub/SFR-Embedding-Code-2B_R', trust_remote_code=True,
                                  low_cpu_mem_usage=False).to(device)

root_dir = "/Users/01428674/Desktop/test"
for root, dirs, files in os.walk(root_dir):
    for file in files:
        file_path = os.path.join(root, file)
        this_root, ext = os.path.splitext(file_path)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                split_contents = chunk_code(content, file_path)
                if len(split_contents) == 0:
                    continue
                vector_list = []
                for i in range(0, len(split_contents), 2):
                    sub_vector_list = model.encode_corpus(split_contents[i:i + 2], max_length=max_length).tolist()
                    vector_list.extend(sub_vector_list)

                # 构造插入lancedb的多行数据
                data = []
                for index in range(len(split_contents)):
                    data.append(
                        {"file_name": file_path, "text": split_contents[index],
                         "chunk_no": index, "vector": vector_list[index]}
                    )
                # 实现增量嵌入
                table.merge_insert(["file_name", "chunk_no"]) \
                    .when_not_matched_insert_all() \
                    .when_matched_update_all() \
                    .when_not_matched_by_source_delete() \
                    .execute(data)

        except UnicodeDecodeError:
            # 处理非文本文件（如图片、二进制文件）
            with open(file_path, 'rb') as f:
                binary_data = f.read()
                print(f"二进制文件: {file_path}, 大小: {len(binary_data)}字节")
        except Exception as e:
            print(f"读取失败: {file_path}, 错误: {e}")

table_pandas = table.to_pandas()
print(table_pandas)

query_instruction_example = "given code, retrieval relevant content"
queries = [
    "how to get group dept codes by a dept code?"
]
query_embeddings = model.encode_queries(queries, instruction=query_instruction_example, max_length=max_length)
query_embeddings_list = query_embeddings.tolist()

results = table.search(query_embeddings_list[0]).limit(4).to_list()
print(results)

queries = [
    "如何根据网点编码获取父子网点组?"
]
query_embeddings = model.encode_queries(queries, instruction=query_instruction_example, max_length=max_length)
query_embeddings_list = query_embeddings.tolist()
results = table.search(query_embeddings_list[0]).limit(4).to_list()
print(results)
