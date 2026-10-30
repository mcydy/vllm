from vllm import LLM, SamplingParams

MODEL_PATH = "/home/jeff/work/demo/Qwen3-4B-Instruct-2507"

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=256,
)

llm = LLM(
    model=MODEL_PATH,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.85,
    max_model_len=8192,
    trust_remote_code=False,
    enforce_eager=False,   # 调试时可先设 True 关 CUDA Graph
)

prompts = [
    "hello,what's you name?",
    "用一句话解释什么是 PagedAttention。",
    "用一句话解释什么是 RadixAttention",
]

outputs = llm.generate(prompts, sampling_params)
for o in outputs:
    print("prompy:", o.prompt)
    print("output:", o.outputs[0].text)
    print("-" * 60)