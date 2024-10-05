import numpy as np
#https://huggingface.co/metrics
"""
mrc_preds: {
    predictions: [{
        id: str
        prediction_text: str
    }] //len=(# of data point)
    label_ids: [{
        id: str
        answers: {
            answer_start: list
            text: list
        }
    }] //len=(# of data point)
}
retrieval_preds: {
    sim_score: np.array, //shape=(# of data point, num_neg+1)
    targets: np.array, //shape=(# of data point)
}
"""

def squad(mrc_preds, method):
    predictions = mrc_preds.predictions
    references = mrc_preds.label_ids
    result = method.compute(predictions=predictions, references=references)

    return {'exact_match': result['exact_match'], 'f1': result['f1']}

    return result

def bleu(mrc_preds, method):
    predictions = [pred["prediction_text"] for pred in mrc_preds.predictions]
    references = [label["answers"]['text'] for label in mrc_preds.label_ids]
    result = method.compute(predictions=predictions, references=references)

    return {'bleu': result['bleu']}

def accuracy(retrieval_preds, method):
    predictions = np.argmax(retrieval_preds['sim_score'], axis=-1).tolist()
    references = retrieval_preds['targets'].tolist()
    result = method.compute(predictions=predictions, references=references)

    return {'accuracy': result['accuracy']}

def f1(retrieval_preds, method):
    predictions = np.argmax(retrieval_preds['sim_score'], axis=-1).tolist()
    references = retrieval_preds['targets'].tolist()
    result = method.compute(predictions=predictions, references=references, average=None)

    return {'f1': result['f1'][0]}