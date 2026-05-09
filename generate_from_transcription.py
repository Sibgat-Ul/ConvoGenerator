import asyncio
from openai import AsyncOpenAI
from datasets import load_dataset, Dataset, concatenate_datasets

ENDPOINTS = [
    "https://zjni96p7tqvly3-8000.proxy.runpod.net/v1",
    "https://wf8p1hkcb5at6q-8000.proxy.runpod.net/v1",
    "http://10.102.35.205:8000/v1",
]
MODEL = "gemma_3_4"
CONCURRENCY_PER_ENDPOINT = 32 

ds = load_dataset("json", data_files="to_transcript.json", split="train")

shards = [
    ds.shard(num_shards=len(ENDPOINTS), index=i)
    for i in range(len(ENDPOINTS))
]

async def process_example(client, semaphore, example):
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": example["transcription"]}],
                max_tokens=512,
                temperature=0.0,
            )
            example["lm_response"] = response.choices[0].message.content.strip()
        except Exception as e:
            example["lm_response"] = None
            example["error"] = str(e)
        return example

async def worker(endpoint, shard, output_file):
    client = AsyncOpenAI(base_url=endpoint, api_key="sk-xxxxxxxx")
    semaphore = asyncio.Semaphore(CONCURRENCY_PER_ENDPOINT)
    tasks = [process_example(client, semaphore, dict(ex)) for ex in shard]
    results = await asyncio.gather(*tasks)
    Dataset.from_list(list(results)).to_json(output_file)

async def main():
    await asyncio.gather(*[
        worker(endpoint, shards[i], f"shard_{i}.json")
        for i, endpoint in enumerate(ENDPOINTS)
    ])

    merged = load_dataset(
        "json",
        data_files=[f"shard_{i}.json" for i in range(len(ENDPOINTS))],
        split="train"
    )
    merged.to_json("generated_completions.json")

asyncio.run(main())