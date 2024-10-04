import sys, os
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict, Sequence, Features, Value
"""
STANDARD DATA FORMAT
{
    'id': Value(dtype='string', id=None),
    'title': Value(dtype='string', id=None),
    'context': Value(dtype='string', id=None),
    'question': Value(dtype='string', id=None),
    'answers': Sequence(feature={
        'text': Value(dtype='string', id=None), 
        'answer_start': Value(dtype='int32', id=None)
    }, length=-1, id=None)
}
"""

def get_standard_features():
    return Features({
        "id": Value("string"),
        "title": Value("string"),
        "context": Value("string"),
        "question": Value("string"),
        "answers": Sequence({
            "text": Value("string"),
            "answer_start": Value("int32")
        })
    })

def get_dataset_list(dataset_name_list):
    parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_list = []
    for dataset_name in dataset_name_list:
        if os.path.exists(f"{parent_directory}/data/{dataset_name}"):
            dataset_list.append(load_from_disk(f"{parent_directory}/data/{dataset_name}"))
        else:
            # download dataset and formatting
            current_module = sys.modules[__name__]
            getattr(current_module, dataset_name)()
            dataset_list.append(load_from_disk(f"{parent_directory}/data/{dataset_name}"))
    return dataset_list

def squad_kor_v1():
    squad_kor_v1_dataset = load_dataset("squad_kor_v1")
    squad_kor_v1_train_dataset = squad_kor_v1_dataset["train"]
    squad_kor_v1_validation_dataset = squad_kor_v1_dataset["validation"]

    train_dataset = Dataset.from_dict({
        "id": [guid for guid in squad_kor_v1_train_dataset["id"]],
        "title": [title for title in squad_kor_v1_train_dataset["title"]],
        "context": [context for context in squad_kor_v1_train_dataset["context"]],
        "question": [question for question in squad_kor_v1_train_dataset["question"]],
        "answers": Dataset.from_dict({
            "text": [answers["text"] for answers in squad_kor_v1_train_dataset["answers"]],
            "answer_start": [answers["answer_start"] for answers in squad_kor_v1_train_dataset["answers"]]
        })
   }, features=get_standard_features())

    validation_dataset = Dataset.from_dict({
        "id": [guid for guid in squad_kor_v1_validation_dataset["id"]],
        "title": [title for title in squad_kor_v1_validation_dataset["title"]],
        "context": [context for context in squad_kor_v1_validation_dataset["context"]],
        "question": [question for question in squad_kor_v1_validation_dataset["question"]],
        "answers": Dataset.from_dict({
            "text": [answers["text"] for answers in squad_kor_v1_validation_dataset["answers"]],
            "answer_start": [answers["answer_start"] for answers in squad_kor_v1_validation_dataset["answers"]]
        })
   }, features=get_standard_features())

    final_dataset = DatasetDict({
        "train": train_dataset,
        "validation": validation_dataset
    })

    parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.exists(f"{parent_directory}/data/"):
        os.makedirs(f"{parent_directory}/data/")
    final_dataset.save_to_disk(f"{parent_directory}/data/squad_kor_v1")

def klue_mrc():
    klue_mrc_dataset = load_dataset("klue", "mrc")
    klue_mrc_train_dataset = klue_mrc_dataset["train"]
    klue_mrc_validation_dataset = klue_mrc_dataset["validation"]

    '''
    train_dataset = Dataset.from_dict({
        "id": klue_mrc_train_dataset["guid"],
        "title": klue_mrc_train_dataset["title"],
        "context": klue_mrc_train_dataset["context"],
        "question": klue_mrc_train_dataset["question"],
        "answers": klue_mrc_train_dataset["answers"],
   }, features=get_standard_features())

    validation_dataset = Dataset.from_dict({
        "id": klue_mrc_validation_dataset["guid"],
        "title": klue_mrc_validation_dataset["title"],
        "context": klue_mrc_validation_dataset["context"],
        "question": klue_mrc_validation_dataset["question"],
        "answers": klue_mrc_validation_dataset["answers"],
   }, features=get_standard_features())
   '''

    train_dataset = Dataset.from_dict({
        "id": [guid for guid in klue_mrc_train_dataset["guid"]],
        "title": [title for title in klue_mrc_train_dataset["title"]],
        "context": [context for context in klue_mrc_train_dataset["context"]],
        "question": [question for question in klue_mrc_train_dataset["question"]],
        "answers": Dataset.from_dict({
            "text": [answers["text"] for answers in klue_mrc_train_dataset["answers"]],
            "answer_start": [answers["answer_start"] for answers in klue_mrc_train_dataset["answers"]]
        })
   }, features=get_standard_features())

    validation_dataset = Dataset.from_dict({
        "id": [guid for guid in klue_mrc_validation_dataset["guid"]],
        "title": [title for title in klue_mrc_validation_dataset["title"]],
        "context": [context for context in klue_mrc_validation_dataset["context"]],
        "question": [question for question in klue_mrc_validation_dataset["question"]],
        "answers": Dataset.from_dict({
            "text": [answers["text"] for answers in klue_mrc_validation_dataset["answers"]],
            "answer_start": [answers["answer_start"] for answers in klue_mrc_validation_dataset["answers"]]
        })
   }, features=get_standard_features())

    final_dataset = DatasetDict({
        "train": train_dataset,
        "validation": validation_dataset
    })

    parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.exists(f"{parent_directory}/data/"):
        os.makedirs(f"{parent_directory}/data/")
    final_dataset.save_to_disk(f"{parent_directory}/data/klue_mrc")

if __name__=="__main__":
    klue_mrc()