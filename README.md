# Aalap: A 32K context length Indian legal LLM

Aalap (Assistant for Legal and Paralegal functions in India) is an instructions fine-tuned version of Mistral 7B that can perform specific legal tasks in the Indian context. The details about which specific legal tasks Aalap can perform and training dataset can be found [here](https://huggingface.co/datasets/opennyaiorg/aalap_instruction_dataset).

This research model intends to show that we can develop tasks for the legal domain and teach LLMs to do them at an affordable cost.
The details about the dataset, model training, and evaluations can be found in the [paper](). Aalap training & evaluation code can be found on the [repo](https://github.com/OpenNyAI/aalap_legal_llm). 
The performance of Aalap is better than gpt-3.5-turbo in 31% of our test data and obtains an equivalent score in 34% of the test data as evaluated by GPT4.

## What is Aalap’s intended use(s)?
From the evaluation results, we can conclude that, for the tasks that are present in the training data, Aalap is performing comparably to ‘gpt-3.5-turbo’. But for the AIBE exam and Legalbench data, Aalap is not doing any better than the Mistral 7B base model. Hence, Aalap is not a general-purpose India legal LLM but can do well under the constraints of specific legal tasks.

%# Try hosted Aalap
%You can try the hosted version of Aalap [here]().

# Host your own version of Aalap
## Environment Set-up 
To set up the environment, we use conda. Please install anaconda3 or miniconda and then run the following command:

```sh
conda create -n llm_train python==3.11.6
conda activate llm_train
```

```sh
bash enviroment_setup.sh
```

## Fine-tuning
### B-float16 LORA fine-tuning for Aalap
For the fine-tuning process we use deepspeed setup. We completed our fine-tuning over 4xA100 80GB GPU's. And total fine-tuning took around 88hrs to complete.

All the hyperparameters are set in the `train_aalap.sh` file. To start the fine-tuning process run the following command:
```sh
bash train_aalap.sh
```
### 4-bit LORA fine-tuning for Aalap
For experimental purposes, a notebook is provided for 4-bit LORA fine-tuning [here](notebooks/Llama2_4bit_finetune.ipynb).

***Note**: Training all layers with LORA has proven effective, offering comparable performance to fully fine-tuned models in less time and resources.*

## Creating full model from adapter weights
Refer to the [notebook](notebooks/llm_combine_and_save.ipynb) to create a full model from adapter weights.

## Deployment
Deploying Aalap is facilitated using TGI developed by Hugging Face. Use the following command, ensuring you have a Hugging Face account and a created token. You can create a token from [here](https://huggingface.co/settings/token).

```sh
docker run -d --gpus 1,2,3,4 -it --cpus="20" --restart always --nameaalap -v $PWD/temp_models_data:/data -p 8080:80  --shm-size 8g -e HUGGING_FACE_HUB_TOKEN=[YOUR HUGGING FACE TOKEN] -e SHARDED=true -e MAX_INPUT_LENGTH=32766 -e MAX_TOTAL_TOKENS=65536 -e HOSTNAME=0.0.0.0 -e MODEL_ID=opennyaiorg/Aalap-Mistral-7B-v0.1-bf16 ghcr.io/huggingface/text-generation-inference --model-id opennyaiorg/Aalap-Mistral-7B-v0.1-bf16
```

# Evaluation
The aalap model was evaluated using three methodologies.
1.  Using GPT4 as an evaluator on Aalap test data
2. Using Legalbench data
3. Using All India Bar Exam (AIBE) data

Please refer to the [paper]() for details. Evaluation scripts used to assess the model are included [here](evaluation/README.md).

***Note:** Some evaluation scripts utilize a TGI server to generate responses for different models such as Aalap or Mistral.*

# Citation
If you use this model in your work, please cite the following paper:

```bibtex
```
