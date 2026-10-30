import asyncio
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.sampling_params import SamplingParams

engine_args = AsyncEngineArgs(
    model="/home/jeff/work/demo/Qwen3-4B-Instruct-2507",
    max_model_len=8192,
    gpu_memory_utilization=0.85,
    enforce_eager=True,
)

engine: AsyncLLM = AsyncLLM.from_engine_args(engine_args)

async def main():
    sp = SamplingParams(max_tokens=64, temperature=0.7, top_p=0.9)
    gen = engine.generate(
        "天空为什么是蓝色的",
        sp,
        request_id="req-1",
    )
    async for out in gen:
        final = out
    print(final.outputs[0].text)

asyncio.run(main())