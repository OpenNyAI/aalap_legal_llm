import re

from datasets import load_dataset
from openai import OpenAI
from text_generation import Client
from tqdm.notebook import tqdm

openai_client = OpenAI()
endpoint_url = 'http://localhost:8080'  # URL for text generation inference endpoint
tgi_client = Client(endpoint_url, timeout=10000)
MAX_GENERATION_LENGTH = 3096
OPENAI_MODEL_NAME = "gpt-3.5-turbo-1106"
TEMPERATURE = 0.01


def get_open_ai_response(system_prompt, user_prompt, MAX_GENERATION_LENGTH=MAX_GENERATION_LENGTH,
                         OPENAI_MODEL_NAME=OPENAI_MODEL_NAME, TEMPERATURE=TEMPERATURE):
    response = openai_client.chat.completions.create(model=OPENAI_MODEL_NAME,
                                                     messages=[
                                                         {"role": "system", "content": system_prompt},
                                                         {"role": "user", "content": user_prompt},
                                                     ],
                                                     max_tokens=MAX_GENERATION_LENGTH, temperature=TEMPERATURE
                                                     )
    return response.choices[0].message.content


def get_tgi_response(prompt, MAX_GENERATION_LENGTH=MAX_GENERATION_LENGTH, TEMPERATURE=TEMPERATURE):
    generated_text = tgi_client.generate(prompt,
                                         max_new_tokens=MAX_GENERATION_LENGTH,
                                         repetition_penalty=1.01,
                                         return_full_text=False,
                                         stop_sequences=["</s>"],
                                         temperature=TEMPERATURE,
                                         top_k=10,
                                         top_n_tokens=5,
                                         top_p=0.95,
                                         typical_p=0.95).generated_text
    return generated_text


if __name__ == '__main__':
    test_dataset = load_dataset("opennyaiorg/aalap_instruction_dataset", split="test").to_pandas()
    test_dataset.fillna("", inplace=True)
    if "gpt_3.5_response" not in test_dataset.columns:
        test_dataset["gpt_3.5_response"] = None

    if "aalap_response" not in test_dataset.columns:
        test_dataset["aalap_response"] = None

    if "mistral_7b_response" not in test_dataset.columns:
        test_dataset["mistral_7b_response"] = None

    for index, row in tqdm(test_dataset.iterrows(), total=len(test_dataset), desc="Generating GPT 3.5 responses"):
        if test_dataset.loc[index, "gpt_3.5_response"] is None:
            system_prompt = row["system_prompt"] if row["system_prompt"] else ''
            user_prompt = row["user_prompt"] if row["user_prompt"] else ''
            input_text = row["input_text"] if row["input_text"] else ''
            user_prompt = user_prompt + '\n' + input_text
            try:
                response = get_open_ai_response(system_prompt, user_prompt)
            except:
                response = get_open_ai_response(system_prompt, user_prompt, MAX_GENERATION_LENGTH=512)
            test_dataset.loc[index, "gpt_3.5_response"] = response

    for model_being_used in ["mistral_7b_response", "aalap_response"]:
        for index, row in tqdm(test_dataset.iterrows(), total=len(test_dataset),
                               desc=f"Generating TGI responses for {model_being_used}"):
            if test_dataset.loc[index, model_being_used] is None:
                system_prompt = row["system_prompt"] if row["system_prompt"] else ''
                user_prompt = row["user_prompt"] if row["user_prompt"] else ''
                input_text = row["input_text"] if row["input_text"] else ''
                user_prompt = user_prompt + '\n' + input_text
                if model_being_used == "mistral_7b_response":
                    prompt = f"{system_prompt} {user_prompt}"
                    max_length = 1024
                else:
                    prompt = f"<s> [INST] <<SYS>> {system_prompt} <</SYS>> {user_prompt} [/INST]"
                    max_length = MAX_GENERATION_LENGTH
                response = get_tgi_response(prompt=prompt, MAX_GENERATION_LENGTH=max_length)
                if model_being_used == "aalap_response":
                    response = re.sub(r'^[^:]+:', '', response.strip()).strip()
                test_dataset.loc[index, model_being_used] = response
        print(f"Completed generating responses for {model_being_used}")
        input("Press enter to continue to next model")

    test_dataset.to_csv("aalap_test_data_results.csv", index=False)
