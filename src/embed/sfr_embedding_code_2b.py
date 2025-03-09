import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# Each query needs to be accompanied by an corresponding instruction describing the task.
query_instruction_example = "Given Code or Text, retrieval relevant content"
queries = [
    "how to implement quick sort in Python?"
]

# No instruction needed for retrieval passages
passages = [
    "def quick_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quick_sort(left) + middle + quick_sort(right)",
    "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n-i-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]\n    return arr"
]

device = torch.device("mps")
# load model with tokenizer
model = AutoModel.from_pretrained('/Users/01428674/Ai/model-hub/SFR-Embedding-Code-2B_R', trust_remote_code=True, low_cpu_mem_usage=False).to(device)
# model = model.to_empty(device=device)  # 从 meta 设备转移到目标设备并分配空内存
# model.load_state_dict(model.state_dict())  # 重新初始化参数（若需要可加载预训练权重）

# get the embeddings
max_length = 1024
query_embeddings = model.encode_queries(queries, instruction=query_instruction_example, max_length=max_length)
passage_embeddings = model.encode_corpus(passages, max_length=max_length)

# normalize embeddings
query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
passage_embeddings = F.normalize(passage_embeddings, p=2, dim=1)

scores = (query_embeddings @ passage_embeddings.T) * 100
print(scores.tolist())
