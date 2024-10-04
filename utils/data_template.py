import sys, os
from datasets import load_dataset, Dataset, DatasetDict, Sequence, Features, Value
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