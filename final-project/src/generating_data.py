import json
import random

import openai
import tiktoken
from datasets import load_dataset
from p_tqdm import p_map
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
from tqdm.auto import tqdm

# API keys
openai.api_key = "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"
random.seed(2)


def format_output(rationale):
    return f"Rationale: {rationale}"


def prepare_user_question(question):
    return f"Question: {question}"


def prepare_user_answer(answer):
    return f"Answer: {answer}"


def prepare_message(question, rationale, answer=None):
    messages = [{"role": "user", "content": prepare_user_question(question)}]
    if answer is not None:
        messages.append({"role": "user", "content": prepare_user_answer(answer)})
    messages.append({"role": "assistant", "content": format_output(rationale)})
    return messages


def get_openai_response(messages):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.8)

    return response["choices"][0]["message"]["content"]


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(messages):
    return get_openai_response(messages)


def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    enc = tiktoken.get_encoding("cl100k_base")
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    num_tokens = len(encoding.encode(string))
    return num_tokens


def generate_rationales(data, few_shot_examples, output_file):
    system_instruction = "Act as a human teacher who generates a rationale to answer each question."
    
    base_messages = [{"role": "system", "content": system_instruction}]

    for example in few_shot_examples:
        question = example['question']
        rationale = example['rationale']

        base_messages.extend(prepare_message(question, rationale))

    with open(output_file, 'a') as fout:
        for i, example in tqdm(enumerate(data), total=len(data), desc='Generating rationales'):
            messages = list(base_messages)
            
            messages.append({
                "role": "user", "content": 
                prepare_user_question(example['question'])
            })
            rationale = completion_with_backoff(messages)        

            if rationale.lower().startswith("rationale:"):
                rationale = rationale[len("rationale:"):].strip()

            output = {
                'rationale': rationale, 
                'question': example['question'], 
                'answer': example['answer'], 
                'true_rationale': example['rationale']}

            fout.write(json.dumps(output) + '\n')
            if i % 1000 == 0:
                print(output)


if __name__ == '__main__':
    # Example usage
    cot_dataset = load_dataset('csv', data_files='../data/cot_train_data.csv')
    cot_dataset = cot_dataset.shuffle()

    # Create few-shot examples
    sample_indices = list(range(5))
    few_shot_examples = cot_dataset['train'].select(sample_indices)

    # Remove the sampled examples from the dataset
    cot_dataset['train'] = cot_dataset['train'].filter(
        lambda example: example['question'] not in few_shot_examples['question']
    )
    initial_idx = 0
    final_idx = 2500
    generate_rationales(cot_dataset['train'].select(range(initial_idx, final_idx)), few_shot_examples, 
                        f"../data/rationales_generated_by_chatgpt_{initial_idx}_{final_idx}.jsonl")
    
