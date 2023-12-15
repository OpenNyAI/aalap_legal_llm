# [Aalap a 32K context length legal LLM](https://huggingface.co/opennyaiorg/Aalap-Mistral-7B-v0.1-bf16)

Aalap(Assistant for Legal and Paralegal functions in India) is built for research purposes only and provides a single turn response in tasks such as generating legal explanation and perform legal tasks over user given details and question. For more details about the task model is capable of checkout the [dataset](https://huggingface.co/datasets/opennyaiorg/aalap_instruction_dataset) used for training the model. The model is designed to excel particularly in generating explanations.

Note that:

1. This is a research model, intended to show that we can develop task for legal domain and teach SLM's to do them.
2. We have synthetically generated data for various legal tasks and used a portion of ORCA dataset relevant to law for teach LLM do explanation and certain tasks.
3. The model is not optimized for chat and has not been trained with RLHF or DPO. It can sometimes generate wrong information.
4. Beyond explanation generation, the model inherits capabilities and limitations of its base (Mistral 7B).

We make [Aalap](https://huggingface.co/opennyaiorg/Aalap-Mistral-7B-v0.1-bf16) weights publicly available to support further research on the development.


# Environment Set-up 
To set-up the environment we use conda. Please install anaconda3 or miniconda and then run following command:

```sh
conda create -n llm_train python==3.11.6
conda activate llm_train
```

```sh
bash enviroment_setup.sh
```

# Fine-tuning
### B-float16 LORA fine-tuning for Aalap
For the fine-tuning process we use deepspeed setup. We completed our fine-tuning over 4xA100 80GB GPU's. And total fine-tuning took around 88hrs to complete.

All the hyperparameters are set in the `train_aalap.sh` file. To start the fine-tuning process run the following command:
```sh
bash train_aalap.sh
```
### 4-bit LORA fine-tuning for Aalap
For experimental purposes, a notebook is provided for 4-bit LORA fine-tuning [here](notebooks/Llama2_4bit_finetune.ipynb).

***Note**: Training all layers with LORA has proven effective, offering comparable performance to fully fine-tuned models in less time and resources.*

# Creating full model from adapter weights
Refer to the [notebook](notebooks/llm_combine_and_save.ipynb) to create a full model from adapter weights.

# Deployment
Deploying Aalap is facilitated using TGI developed by Hugging Face. Use the following command, ensuring you have a Hugging Face account and a created token. You can create a token from [here](https://huggingface.co/settings/token).

```sh
docker run -d --gpus 1,2,3,4 -it --cpus="20" --restart always --nameaalap -v $PWD/temp_models_data:/data -p 8080:80  --shm-size 8g -e HUGGING_FACE_HUB_TOKEN=[YOUR HUGGING FACE TOKEN] -e SHARDED=true -e MAX_INPUT_LENGTH=32766 -e MAX_TOTAL_TOKENS=65536 -e HOSTNAME=0.0.0.0 -e MODEL_ID=opennyaiorg/Aalap-Mistral-7B-v0.1-bf16 ghcr.io/huggingface/text-generation-inference --model-id opennyaiorg/Aalap-Mistral-7B-v0.1-bf16
```

# Evaluation

Various datasets and methods are employed for evaluation; refer to the paper for details. Evaluation scripts used to assess the model are included [here](evaluation/README.md).

***Note:** Some evaluation scripts utilize a TGI server to generate responses for different models such as Aalap or Mistral.*

# Citation
If you use this model in your work, please cite the following paper:

```bibtex
```
