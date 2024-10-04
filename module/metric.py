#https://huggingface.co/metrics

def squad(eval_preds):
    predictions = eval_preds.predictions
    references = eval_preds.label_ids

    return predictions, references

def bleu(eval_preds):
    predictions = [pred["prediction_text"] for pred in eval_preds.predictions]
    references = [label["answers"]['text'] for label in eval_preds.label_ids]

    return predictions, references