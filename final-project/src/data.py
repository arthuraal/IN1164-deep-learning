import copy
from datasets import load_dataset


def tokenize(data, tokenizer):
    encoded_inp = tokenizer(
        data['prompt'],
        truncation=True,
        max_length=256,
        padding=False
    )

    if (encoded_inp['input_ids'][-1] != tokenizer.eos_token_id) and (len(encoded_inp['input_ids']) < 256):
        encoded_inp['input_ids'].append(tokenizer.eos_token_id)
        encoded_inp['attention_mask'].append(1)

    encoded_inp['labels'] = copy.deepcopy(encoded_inp['input_ids'])
    return encoded_inp


if __name__ == '__main__':
    data = load_dataset('csv', data_files='../data/cot_train_data.csv')
    data = data["train"].train_test_split(test_size=0.1, shuffle=True, seed=42)
    print(data)
