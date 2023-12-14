import json
import os.path
import re

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from utils import call_openai_api


def get_gpt4_evaluation_rating(output_dict, model_name):
    candidate_answer = output_dict[model_name + '_response']

    if str(candidate_answer).strip() == '':
        gpt4_evaluation_rating = 0
        gpt4_evaluation_rating_explanation = 'No answer provided'
    else:
        system_prompt = '''Please act as an impartial judge and evaluate the quality of the response provided by a legal AI assistant to the user question. Your evaluation should consider correctness and helpfulness. You will be given a reference answer and the assistant's answer. Begin your evaluation by comparing the assistant's answer with the reference answer. Identify and correct any mistakes. Be as objective as possible. Provide your answer in 300 words. If assistant's answer is not relevant to the question or answer is not provided then give rating as 0. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".'''
        task_description = ''
        if not pd.isna(output_dict['system_prompt']):
            task_description = task_description + output_dict['system_prompt']
        if not pd.isna(output_dict['user_prompt']):
            task_description = task_description + output_dict['user_prompt']
        if not pd.isna(output_dict['input_text']):
            task_description = task_description + output_dict['input_text']

        reference_answer = output_dict['output_text']
        user_prompt = task_description + '\n\n\n<|The Start of Reference Answer|>\n\n' + str(
            reference_answer) + '\n<|The End of Reference Answer|>\n' + '\n\n\n<|The Start of assistant\'s answer to evaluate|>\n\n' + str(
            candidate_answer) + '\n<|The End of assistant\'s answer to evaluate|>'
        messages = [
            {"role": "system",
             "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        gpt4_evaluation_rating_explanation = call_openai_api(messages, max_tokens=800, model='gpt-4-1106-preview')
        try:
            gpt4_evaluation_rating = re.search(r'\[\[\d+\]\]', gpt4_evaluation_rating_explanation).group(0).replace(
                '[[', '').replace(']]', '')
        except:
            gpt4_evaluation_rating = None

    return {model_name + '_evaluation_rating_gpt4': gpt4_evaluation_rating,
            model_name + '_evaluation_rating_explanation_gpt4': gpt4_evaluation_rating_explanation}


def get_gpt4_evaluation_rating_for_all_models(row):
    row, row_number, intermediate_output_path = row['row'], row['row_number'], row['intermediate_output_path']
    output_file_path = os.path.join(intermediate_output_path, str(row_number) + '.json')
    if os.path.exists(output_file_path):
        row = json.load(open(output_file_path))

    models = ['aalap', 'mistral_7b', 'gpt_3.5']
    for model in models:
        if row.get(model + '_evaluation_rating_gpt4') is None:
            row.update(get_gpt4_evaluation_rating(row, model_name=model))

            json.dump(row, open(os.path.join(intermediate_output_path, str(row_number) + '.json'), 'w'), indent=4)
    return row


if __name__ == '__main__':
    model_outputs_path = 'aalap_test_data_results.csv'
    intermediate_output_path = 'intermediate'
    os.makedirs(intermediate_output_path, exist_ok=True)
    data = pd.read_csv(model_outputs_path)
    data_records = data.to_dict('records')
    data_records = [{'row': i, 'row_number': id, 'intermediate_output_path': intermediate_output_path} for id, i in
                    enumerate(data_records)]

    result = Parallel(n_jobs=5)(delayed(get_gpt4_evaluation_rating_for_all_models)(i) for i in tqdm(data_records))

    # data.to_csv('/Users/prathamesh/tw_projects/OpenNyAI/data/LLM/eval/aalap_test_data_results_with_evaluations.csv',index=False)
