
def mean_pooling(token_embeddings, attention_mask):
    token_embeddings = token_embeddings.masked_fill(~attention_mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    return sentence_embeddings

def format_data(data):
    return data["title"] + " " + data["text"]