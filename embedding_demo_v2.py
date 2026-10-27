import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np


model_name = "distilbert-base-multilingual-cased"


tokenizer = AutoTokenizer.from_pretrained(model_name)


model = AutoModel.from_pretrained(model_name)


model.eval()


text1 = 'car'
text2 = 'building'
text3 = 'nature'

def get_tokens(string):
    tokens = tokenizer(
        string,
        return_tensors="pt"
    )
    return tokenizer.convert_ids_to_tokens(tokens["input_ids"][0]), tokens

# output, tokens = get_tokens(text1)
# print(output)
# print(tokens)
# print("Tokens:")
# print(get_tokens(text1))

input_string = 'i love programming, coding and swimming'
result = get_tokens(input_string)
print(result)

tokens1 = get_tokens(text1)[1]
tokens2 = get_tokens(text2)[1]
tokens3 = get_tokens(text3)[1]

def get_embedding(tokens):
    with torch.no_grad():
        outputs = model(**tokens)

    embeddings = outputs.last_hidden_state
    return embeddings

embedding = get_embedding(result[1])
print(embedding.numpy().squeeze().shape)
# emb = get_embedding(tokens)[0]
# print('<----->')
# print(emb)
# print(emb.shape)

# def get_sentence_embedding(tokens):
#     attention_mask = tokens["attention_mask"]

#     with torch.no_grad():
#         outputs = model(**tokens)

#     token_embeddings = outputs.last_hidden_state

#     mask = attention_mask.unsqueeze(-1)

#     sentence_embedding = (
#         token_embeddings * mask
#     ).sum(dim=1) / mask.sum(dim=1)
#     return sentence_embedding

# print("\nEmbeddings shape:")
emb1 = get_embedding(tokens1)[0, 1].numpy()
emb2 = get_embedding(tokens2)[0, 1].numpy()
emb3 = get_embedding(tokens3)[0, 1].numpy()

# print(tokens1)
# print(emb1)

def get_similarity(vector1, vector2):
    return np.dot(vector1, vector2)/(np.linalg.norm(vector1)*np.linalg.norm(vector2))

progr_token = embedding.numpy().squeeze()[3, :]
swim_token = embedding.numpy().squeeze()[-2, :]
print(result[1]['input_ids'][0, 3])
print(result[1]['input_ids'][0, -2])

similarity = get_similarity(progr_token, swim_token)
print(similarity)




cos12 = get_similarity(emb1, emb2)
cos13 = get_similarity(emb1, emb3)

v1 = np.array([-2, -2])
v2 = np.array([5, 5])
cosv1v2 = get_similarity(v1, v2)
# print(cos12)
# print(cos13)
print(cosv1v2)


# sentence_emb1 = get_sentence_embedding(tokens1)
# print(sentence_emb1)