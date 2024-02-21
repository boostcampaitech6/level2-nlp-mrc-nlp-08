from datasets import load_from_disk, DatasetDict

# 경로 설정
original_dataset = load_from_disk("/data/ephemeral/data/train_dataset/")

# print(original_dataset['train'])

# print(type(original_dataset['train']))
# print(original_dataset['train'][0])

def modify_question(example):
    example['question'] += " [MASK]"  # Replace 'question' with " [MASK]"
    return example

modified_dataset = original_dataset['train'].map(modify_question)

# Create a new DatasetDict with the modified 'train' subset
modified_dataset_dict = {
    'train': modified_dataset,
    'validation': original_dataset['validation'],
}

print(modified_dataset_dict['train'][0])

#print(modified_dataset_dict['train'][0])

modified_dataset = DatasetDict(modified_dataset_dict)

#print(modified_dataset)
modified_dataset.save_to_disk("/data/ephemeral/data/train_dataset_modified/")