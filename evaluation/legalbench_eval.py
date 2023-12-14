import copy
import random
from datasets import load_dataset, get_dataset_config_names
from text_generation import Client
import os
import json
from tqdm import tqdm
from joblib import Parallel, delayed
from utils import call_openai_api
import re
import pandas as pd
from pqdm.processes import pqdm
import time
class LegalbenchEval:
    def __init__(self, ip_address='localhost', port='8080'):
        endpoint_url = 'http://'+ip_address+ ':' + port +'/'
        self.dataset_name = "nguha/legalbench"
        self.llm_client = Client(endpoint_url, timeout=1000)
        self.legalbench_tasks_path = '/Users/prathamesh/tw_projects/OpenNyAI/git_repos/legal_llm_data_pipelines/src/preannotations/legalbench/tasks'
        self.experiment_name = 'aalap_1024_t_0_1'
        self.output_path = '/Users/prathamesh/tw_projects/OpenNyAI/data/LLM/eval/'+self.experiment_name
        self.test_input_data_path = '/Users/prathamesh/tw_projects/OpenNyAI/data/LLM/eval/sample_test_input_data.json'

        self.system_prompts_concise = [
            'You are a helpful assistant. Answer the following question with a short answer. Dont explain too much.'
            "You're a valuable assistant. Provide a concise answer to the following question without explanation.",
            "As a helpful assistant, please give a brief response to the following question without elaboration.",
            "You are an assisting hand. Offer a short answer to the following question without any explanation.",
            "In your role as a helpful assistant, respond briefly to the following question without providing an explanation.",
            "You serve as a useful assistant. Give a concise answer to the following question without further elaboration.",
            "You are a supportive assistant. Please answer the following question briefly and avoid any explanations.",
            "As an assistant, give a short response to the following question without any explanation.",
            "In your capacity as a helpful assistant, provide a brief answer to the following question without elaboration.",
            "You act as a helpful assistant. Respond to the following question with a short answer and avoid explanations.",
            "You're a great assistant. Answer the following question briefly, and refrain from explaining the answer."
        ]

        ### create test input data if it does not exist
        if not os.path.exists(self.test_input_data_path):
            test_input_data = self._create_test_input_data()
            json.dump(test_input_data,open(self.test_input_data_path,'w'),indent=4)

        self.test_input_data = json.load(open(self.test_input_data_path,'r'))
        self.test_input_data = self._create_task_samples(per_task_sample_size=10)

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path,exist_ok=True)

    def _create_test_input_data(self):
        test_input_data = []
        tasks = get_dataset_config_names(self.dataset_name)
        for task in tqdm(tasks):
            try:
                base_prompt = open(os.path.join(self.legalbench_tasks_path, task, "base_prompt.txt")).read()
            except:
                base_prompt = open(os.path.join(self.legalbench_tasks_path, task, "claude_prompt.txt")).read()

            test_data = load_dataset(self.dataset_name, task, split='test')
            random.seed(10)

            training_prompts = self.generate_prompts(base_prompt, test_data)
            training_prompts = random.sample(training_prompts, min(100, len(test_data)))

            for training_prompt in training_prompts:
                training_prompt['task'] = 'legalbench'
                training_prompt['subtask'] = task
                training_prompt['system_prompt'] = random.choice(self.system_prompts_concise)

            test_input_data.extend(training_prompts)
        return test_input_data
    def generate_prompts(self, prompt_template: str, test_data) -> list[dict]:
        prompts = []
        for dd in test_data:
            #prompt = self.explain_answer_prompt + '\n' + str(prompt_template)
            prompt = str(prompt_template)
            for k, v in dd.items():
                if k != "answer" and k != "index":
                    prompt = prompt.replace("{{" + k + "}}", str(v))
                    #prompt = prompt + "\n" + k + ": " + str(v)

            prompts.append({'user_prompt': prompt, 'answer': dd.get("answer")})
        # assert len(set(prompts)) == len(prompts), "Duplicated prompts detected"
        return prompts

    def call_llm_api(self,system_prompt,user_prompt,max_new_tokens):
        prompt = f"<s>[INST] <<SYS>>{system_prompt}<</SYS>>\n\n{user_prompt}\n[/INST]"
        #prompt = user_prompt
        generated_text = self.llm_client.generate(prompt,
                                         max_new_tokens=max_new_tokens,
                                         repetition_penalty=1.03,
                                         return_full_text=False,
                                         stop_sequences=["</s>"],
                                         temperature=0.1,
                                         top_k=10,
                                         top_n_tokens=5,
                                         top_p=0.95,
                                         typical_p=0.95).generated_text

        return generated_text

    def _get_llm_answer(self,training_prompt):
        llm_answers = copy.deepcopy(training_prompt)
        system_prompt = training_prompt['system_prompt']
        user_prompt = training_prompt['user_prompt']

        generated_explanations = self.call_llm_api(system_prompt, user_prompt,max_new_tokens=1024)
        generated_explanations = re.sub('^\s+Response:\s+','',generated_explanations)
        generated_explanations = generated_explanations.strip()

        llm_answers['output_text'] = generated_explanations
        llm_answers['is_output_correct'] = self._compare_ground_truth_with_answer_regex(generated_text = generated_explanations,
                                                                    ground_truth = llm_answers['answer'])
        llm_answers['chatgpt_assesment'] = self._chatgpt_assement(task_description = llm_answers['user_prompt'],
                                                                  llm_response = generated_explanations,
                                                                  ground_truth = llm_answers['answer'])
        return llm_answers

    def _get_llm_answers_for_a_chunk(self,tasks):
        chunk_answers = []
        chunk_output_file_path = self.output_path+'/chunks/'+str(tasks['chunk_id'])+'.json'
        if not os.path.exists(chunk_output_file_path):
            for task in tasks['chunk_tasks']:
                chunk_answers.append(self._get_llm_answer(task))
            os.makedirs(self.output_path+'/chunks/',exist_ok=True)
            json.dump(chunk_answers,open(chunk_output_file_path,'w'),indent=4)
        else:
            chunk_answers = json.load(open(chunk_output_file_path,'r'))
        return chunk_answers
    def evaluate(self):

        chunk_size = 100
        test_chunks = [{'chunk_id': i,'chunk_tasks':self.test_input_data[i:i + chunk_size]} for i in range(0, len(self.test_input_data), chunk_size)]
        eval_results = pqdm(test_chunks, self._get_llm_answers_for_a_chunk,n_jobs=5)

        combined_eval_results = []
        for chunk_results in eval_results:
            if type(chunk_results) == list:
                for chunk_result in chunk_results:
                    combined_eval_results.append(chunk_result)
        json.dump(combined_eval_results, open(self.output_path+'/eval_results.json', 'w'))
        return combined_eval_results

    def _chatgpt_assement(self,task_description,llm_response,ground_truth):

        #task_description = task_description.replace(self.explain_answer_prompt,'')
        eval_prompt = f'''Task:{task_description}\n\nLLM Response: {llm_response}\n\nGround Truth: {ground_truth}'''
        messages = [
            {"role": "system", "content": 'Your job to to evaluate response generated by LLM against the ground truth answer provided below. Only output one word "correct" or "incorrect". Do not generate any other text.'},
            {"role": "user", "content": eval_prompt}
        ]
        chatgpt_assesment = call_openai_api(messages, max_tokens=20, model='gpt-3.5-turbo')
        if chatgpt_assesment == 'correct':
            return True
        elif chatgpt_assesment == 'incorrect':
            return False
        else:
            return None
    def _compare_ground_truth_with_answer_regex(self,generated_text,ground_truth):
        ## searches for ground truth answer in generated text. Generated text might contain more words as well. Hence if the ground truth is present in generated text then we treat as a match.
        if re.search(r'\b'+ground_truth+r'\b',generated_text,re.IGNORECASE):
            return True
        else:
            return False

    def _create_task_samples(self, per_task_sample_size):
        sampled_data = []
        for task in get_dataset_config_names(self.dataset_name):
            task_data = [d for d in self.test_input_data if d['subtask'] == task]
            sampled_data.extend(random.sample(task_data, min(per_task_sample_size, len(task_data))))
        return sampled_data

    def _get_chatgpt_answer(self,question_dict):
        system_prompt = question_dict['system_prompt']
        user_prompt = question_dict['user_prompt']

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        chatgpt_answer = call_openai_api(messages, max_tokens=10, model='gpt-3.5-turbo')
        return chatgpt_answer
    def _get_chatgpt_answer_and_evaluate(self):
        results = json.load(open(self.output_path+'/eval_results.json','r'))
        for result in tqdm(results):
            if result.get('chatgpt_answer') is None:
                try:
                    result['chatgpt_answer'] = self._get_chatgpt_answer(result)
                    result['is_chatgpt_output_correct'] = True if result['chatgpt_answer'] == result['answer'] else False
                    json.dump(results,open(self.output_path+'/eval_results.json','w'),indent=4)
                except:
                    print('could not get answer from ChatGPT')
                    time.sleep(5)

if __name__ == "__main__":
    l = LegalbenchEval()
    l._get_chatgpt_answer_and_evaluate()