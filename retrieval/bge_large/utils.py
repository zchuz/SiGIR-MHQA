import torch.nn.functional as F

def pooling(model_output, attention_mask=None):
    sentence_embeddings = model_output[0][:, 0]
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings



def format_data(data):
    return data["title"] + " " + data["text"]