# Model Evaluation
This section provides details on evaluating the performance of the Aalap language model using different scripts and datasets. Below are descriptions of each evaluation script:

# 1. GPT-4 evaluations
We have generated responses from GPT-3.5, Mistral 7B and Aalap. We then use GPT-4 as an evaluator to evaluate and compare the responses. Details of the evaluation are provided in the paper.

### Usage:
First you have to generate a csv file of test data with responses from respective models. For this, we have provided a script `aalap_test_data_response_generation.py`. This will create a csv.
Before you run it you have to create a TGI deployment of Aalap and Mistral. 
For TGI deployment steps refer to Main readme. 
You can run it as follows:
```
python aalap_test_data_response_generation.py
```

Once you have the csv file, you can run the evaluation script as follows:
```
python gpt4_evaluations.py
``` 
# 2. AIBE evaluations
This script evaluates the performance of the Aalap language model on the AIBE dataset.

# 3. legalbench evaluations
This script evaluates the performance of the Aalap language model on the legalbench dataset.

### Notes:
* Ensure that the necessary datasets are available and properly formatted before running the scripts.
* Results obtained from these evaluations contribute to understanding the strengths and limitations of the Aalap language model in various legal contexts.
* For any additional information or assistance, refer to the paper accompanying these evaluation scripts.