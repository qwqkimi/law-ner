from bert_serving.client import BertClient

def BertEmbedding(data):
    bc=BertClient()
    T_embedding=[]
    T_embedding.append(bc.encode(data))
    return T_embedding