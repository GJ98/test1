import numpy as np
#https://huggingface.co/metrics
"""
mrc_preds: {
    predictions: [{
        id: str
        prediction_text: str
    }, len=(# of data point)]
    label_ids: [{
        id: str
        answers: {
            answer_start: list
            text: list
        }
    }, len=(# of data point)]
}
retriever_preds: {
    sim_score: np.array, shape=(# of data point, num_neg+1)
    targets: np.array, shape=(# of data point)
}
"""

def squad(mrc_preds):
    predictions = mrc_preds.predictions
    references = mrc_preds.label_ids

    return {"predictions": predictions, "references": references}

def bleu(mrc_preds):
    predictions = [pred["prediction_text"] for pred in mrc_preds.predictions]
    references = [label["answers"]['text'] for label in mrc_preds.label_ids]

    return {"predictions": predictions, "references": references}

def accuracy(retriever_preds):
    predictions = np.argmax(retriever_preds['sim_score'], axis=-1).tolist()
    references = retriever_preds['targets'].tolist()

    return {"predictions": predictions, "references": references}

def f1(retriever_preds):
    predictions = np.argmax(retriever_preds['sim_score'], axis=-1).tolist()
    references = retriever_preds['targets'].tolist()

    return {"predictions": predictions, "references": references, "average": None}