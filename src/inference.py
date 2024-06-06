import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification

def model_fn(model_dir):
    """
    Load the model for inference.
    """
    model = BertForSequenceClassification.from_pretrained(model_dir)
    return model

def input_fn(request_body, request_content_type):
    """
    Deserialize the request body into an input object.
    """
    if request_content_type == 'application/json':
        request_data = json.loads(request_body)
        return request_data['input']
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """
    Apply the model to the incoming request.
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    inputs = tokenizer(input_data, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    _, predicted = torch.max(outputs.logits, dim=1)
    return predicted.item()

def output_fn(prediction, content_type):
    """
    Serialize the output from the model into the response format.
    """
    return json.dumps({'prediction': prediction})
