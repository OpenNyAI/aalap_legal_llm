import json
import pandas as pd

if __name__ == '__main__':
    # aalap_results = json.load(open('/Users/prathamesh/tw_projects/OpenNyAI/git_repos/legal_llm_training/evaluation/legalbench_eval_results_mistral_7b_bf16_all.json'))
    # aalap_results_combined = []
    # for i in aalap_results:
    #     aalap_results_combined.extend(i)
    aalap_results_combined = json.load(open('/Users/prathamesh/tw_projects/OpenNyAI/data/LLM/eval/aalap_1024_t_0_1/eval_results.json'))
    aalap_results_df = pd.DataFrame(aalap_results_combined)

    mistral_results = json.load(
        open('/Users/prathamesh/tw_projects/OpenNyAI/data/LLM/eval/mistral_7b_base/eval_results.json'))
    mistral_results_df = pd.DataFrame(mistral_results)
    merged = pd.merge(mistral_results_df, aalap_results_df, how='inner', on='user_prompt',suffixes = ('_mistral','_aalap'))


    task_category_df = pd.read_csv('/Users/prathamesh/tw_projects/OpenNyAI/data/LLM/eval/legalbench_task_categories.csv')
    merged = pd.merge(merged, task_category_df, how='left', left_on='subtask_mistral',right_on='subtask')

    merged.groupby('category')['is_output_correct_aalap'].mean()
    merged.groupby('category')['is_output_correct_mistral'].mean()
