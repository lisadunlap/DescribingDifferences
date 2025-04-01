import json
import logging
import os
import threading
from typing import List

import lmdb
import openai
from openai import OpenAI
# import anthropic

from serve.global_vars import LLM_CACHE_FILE, VICUNA_URL, OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY
from serve.utils_general import get_from_cache, save_to_cache

logging.basicConfig(level=logging.INFO)

if not os.path.exists(LLM_CACHE_FILE):
    os.makedirs(LLM_CACHE_FILE)

llm_cache = lmdb.open(LLM_CACHE_FILE, map_size=int(1e11))
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


def get_llm_output(prompt: str, model: str) -> str:
    api_base = {
        "gpt-3.5-turbo": "https://api.openai.com/v1",
        "gpt-4": "https://api.openai.com/v1",
        "gpt-4o-mini": "https://api.openai.com/v1",
        "gpt-4o": "https://api.openai.com/v1",
        "vicuna": VICUNA_URL,
    }
    # TODO: The 'openai.api_base' option isn't read in the client API. You will need to pass it when you instantiate the client, e.g. 'OpenAI(base_url=api_base[model])'
    # openai.api_base = api_base[model]
    client = OpenAI()

    if "gpt" in model:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    else:
        messages = prompt
    key = json.dumps([model, messages])

    cached_value = get_from_cache(key, llm_cache)
    if cached_value is not None:
        logging.debug(f"LLM Cache Hit")
        return cached_value

    for _ in range(3):
        try:
            if "gpt" in model:
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                )
                response = completion.choices[0].message.content.strip()
            elif model == "vicuna":
                completion = client.chat.completions.create(
                    model="lmsys/vicuna-7b-v1.5",
                    prompt=prompt,
                    max_tokens=256,
                    temperature=0,  # TODO: greedy may not be optimal
                )
                response = completion.choices[0].message.content.strip()
            save_to_cache(key, response, llm_cache)
            return response

        except Exception as e:
            logging.error(f"LLM Error: {e}")
            continue
    return "LLM Error: Cannot get response."


def prompt_differences(captions1: List[str], captions2: List[str]) -> str:
    caption1_concat = "\n".join(
        [f"Image {i + 1}: {caption}" for i, caption in enumerate(captions1)]
    )
    caption2_concat = "\n".join(
        [f"Image {i + 1}: {caption}" for i, caption in enumerate(captions2)]
    )
    prompt = f"""Here are two groups of images:

Group 1:
```
{caption1_concat}
```

Group 2:
```
{caption2_concat}
```

What are the differences between the two groups of images?
Think carefully and summarize each difference in JSON format, such as:
```
{{"difference": several words, "rationale": group 1... while group 2...}}
```
Output JSON only. Do not include any other information.
"""
    return prompt


def get_differences(captions1: List[str], captions2: List[str], model: str) -> str:
    prompt = prompt_differences(captions1, captions2)
    differences = get_llm_output(prompt, model)
    try:
        differences = json.loads(differences)
    except Exception as e:
        logging.error(f"Difference Error: {e}")
    return differences


def test_get_llm_output():
    prompt = "hello"
    model = "gpt-4"
    completion = get_llm_output(prompt, model)
    print(f"{model=}, {completion=}")
    model = "gpt-3.5-turbo"
    completion = get_llm_output(prompt, model)
    print(f"{model=}, {completion=}")
    model = "vicuna"
    completion = get_llm_output(prompt, model)
    print(f"{model=}, {completion=}")


def test_get_llm_output_parallel():
    threads = []

    for _ in range(3):
        thread = threading.Thread(target=test_get_llm_output)
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()


def test_get_differences():
    captions1 = [
        "A cat is sitting on a table",
        "A dog is sitting on a table",
        "A pig is sitting on a table",
    ]
    captions2 = [
        "A cat is sitting on the floor",
        "A dog is sitting on the floor",
        "A pig is sitting on the floor",
    ]
    differences = get_differences(captions1, captions2, "gpt-4")
    print(f"{differences=}")


if __name__ == "__main__":
    test_get_llm_output()
    test_get_llm_output_parallel()
    test_get_differences()
