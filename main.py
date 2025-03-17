from fastapi import FastAPI
from pydantic import BaseModel
from src.translate.tencent_smart_trans import TencentSmartTrans
from src.lancedb.embed_query import EmbeddingQuery
import torch
from transformers import AutoModel


app = FastAPI()
embed_query = EmbeddingQuery()
tmt = TencentSmartTrans()
print("正在初始化CodeXEmbed嵌入模型......")
device = torch.device("cpu")
model = AutoModel.from_pretrained('/Users/01428674/Ai/model-hub/SFR-Embedding-Code-2B_R', trust_remote_code=True,
                                  low_cpu_mem_usage=False).to(device)
print("已完成CodeXEmbed嵌入模型的初始化!")
query_instruction_example = "given code, retrieval relevant content"
max_length = 32768


class ContextProviderInput(BaseModel):
    query: str
    fullInput: str


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/retrieve")
async def create_item(item: ContextProviderInput):
    print(f"Continue传入的原始输入:{item.fullInput}")
    # 将中文输入转成英文
    egText = tmt.translate(item.fullInput)
    print(f"Tmt翻译后的英文输入:{egText}")
    queries = [egText]
    # 生成英文输入对应的嵌入向量
    query_embeddings = model.encode_queries(queries, instruction=query_instruction_example, max_length=max_length)
    query_embeddings_list = query_embeddings.tolist()
    # 去查找lancedb中的最优结果
    results = embed_query.search(query_embeddings_list[0])
    # 按照Continue的要求组装返回值
    context_items = []
    for result in results:
        context_items.append({
            "name": result["file_name"],
            "description": result["file_name"],
            "content": result["text"],
        })

    return context_items
