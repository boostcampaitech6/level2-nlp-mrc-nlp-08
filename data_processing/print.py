from datasets import load_from_disk
from collections import Counter


dataset = load_from_disk("/data/ephemeral/data/train_dataset")

# 구조 확인
print(dataset)

dataset = load_from_disk("/data/ephemeral/data/train_dataset_modified")
print(dataset)

# train 선택
split = 'train'
    
# 확인할 example 수
num_examples_to_print = 1

# example 확인
for i in range(num_examples_to_print):
    print(f"\nExample {i + 1} from {split} split:\n")
    example = dataset[split][i]
    for key, value in example.items():
        print(f"{key}: {value}")