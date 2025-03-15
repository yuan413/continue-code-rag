import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# Each query needs to be accompanied by an corresponding instruction describing the task.
query_instruction_example = "Given Code, retrieval relevant content"
queries = [
    "how to implement quick sort in Python?"
]

# No instruction needed for retrieval passages
passages = [
    "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n-i-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]\n    return arr",
    "def get_detailed_instruct(task_description: str, query: str) -> str:return f'Instruct: {task_description}\nQuery: {query}'",
    "def name_or_path(self) -> str: return getattr(self, "", None)",
    "AutoModel.register(CodeXEmbedConfig, CodeXEmbedModel2B)",
    "CodeXEmbedModel2B.register_for_auto_class(\"AutoModel\")",
    "def quick_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quick_sort(left) + middle + quick_sort(right)",
    "def encode_queries(self, queries: List[str], max_length: int, instruction: str, **kwargs) -> np.ndarray: all_queries = [get_detailed_instruct(instruction, query) for query in queries] return self.encode_text(all_queries, max_length)",
    "def encode_corpus(self, corpus: List[str], max_length: int,**kwargs) -> np.ndarray: return self.encode_text(corpus, max_length)",
    "with torch.no_grad():model_output = self.model(**encoded_input) embeddings = self.last_token_pool(model_output, encoded_input['attention_mask'])",
    "pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]",
    "def last_token_pool(self, model_output, attention_mask):last_hidden_states = model_output.last_hidden_state sequence_lengths = attention_mask.sum(dim=1) - 1 batch_size = last_hidden_states.shape[0] return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]",
    "if not return_dict:output = (pooled_logits,) + transformer_outputs[1:] return ((loss,) + output) if loss is not None else output",
    "def split_content(content, lang):split_contents = CodeSplitter.from_defaults(lang, 20, 6, 1200).split_text(content) print(len(split_contents)) return split_contents",
    "class TextModel(LanceModel): text: str = jina_embed.SourceField() vector: Vector(jina_embed.ndims()) = jina_embed.VectorField()",
    "db = lancedb.connect(\"~/.lancedb-2\") tbl = db.create_table(\"test\", schema=TextModel, mode=\"overwrite\")",
    "db.drop_table(\"test_merge_insert\")",
]

device = torch.device("cpu")
# load model with tokenizer
# with torch.profiler.profile(
#     activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.PrivateUse1]
# ) as prof:
model = AutoModel.from_pretrained('/Users/01428674/Ai/model-hub/SFR-Embedding-Code-2B_R', trust_remote_code=True,
                                  low_cpu_mem_usage=False).to(device)
# model = model.to_empty(device=device)  # 从 meta 设备转移到目标设备并分配空内存
# model.load_state_dict(model.state_dict())  # 重新初始化参数（若需要可加载预训练权重）

# get the embeddings
max_length = 32768
query_token_count = model.count_token(queries)
print(f"查询的token总数为{query_token_count}")
query_embeddings = model.encode_queries(queries, instruction=query_instruction_example, max_length=max_length)
query_embeddings_list = query_embeddings.tolist()
output_token_count = model.count_token(passages)
print(f"输出的token总数为{output_token_count}")
passage_embeddings = model.encode_corpus(passages, max_length=max_length)
passage_embeddings_list = passage_embeddings.tolist()
# print(prof.key_averages().table())

# normalize embeddings
query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
passage_embeddings = F.normalize(passage_embeddings, p=2, dim=1)

scores = (query_embeddings @ passage_embeddings.T) * 100
print(scores.tolist())
