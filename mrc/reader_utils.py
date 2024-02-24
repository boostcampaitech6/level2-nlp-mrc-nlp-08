from datasets import load_from_disk, load_dataset

def prepare_dataset(dataset_name, dataset_path=None):
    """return dataset for mrc reader training

    Args:
        data_type: [wiki, squad_kor_v1]
        train_dataset_name: original dataset 경로

    Returns:
        DatasetDict: dataset
    """
    if dataset_name == 'wiki':
        return load_from_disk(dataset_path)
    
    elif dataset_name == 'squad_kor_v1':
        return load_dataset("squad_kor_v1")