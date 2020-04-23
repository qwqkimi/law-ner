from law_ner.utils import process_data
from law_ner.config import *
from law_ner.model_builder.ner import NER
import numpy as np 
from sklearn.metrics import classification_report

def ner_val(model_name,dict_path=dict_path,summary=True):
    (vocab,chunk_tags)=process_data.load_dict(dict_path)
    model=NER()
    model.build_model(len(vocab), embedding_dim, lstm_units, len(chunk_tags),summary=summary)
    model.compile()
    
    predict_text='因涉嫌犯危险驾驶罪,于2014年2月4日被北京市延庆县公安局羁押,次日被刑事拘留,现羁押于北京市延庆县看守所。'
    model.load_weight(models_path,model_name=model_name)

    s,length=process_data.process_data(predict_text,vocab)
    raw = model.predict(s)[0][-length:]
    result = [np.argmax(row) for row in raw]
    result_tags = [chunk_tags[i] for i in result]

    cl=''

    for s, t in zip(predict_text, result_tags):
        if t in ('B-CL', 'I-CL'):
            cl += ' ' + s if (t == 'B-CL') else s
    print(predict_text)
    print(['CL:' + cl])

def ner_prf(model_name,train_path=train_path,test_path=test_path,dict_path=dict_path,summary=True):
    (vocab,chunk_tags)=process_data.load_dict(dict_path)
    model=NER()
    model.build_model(len(vocab), embedding_dim, lstm_units, len(chunk_tags),summary=summary)
    model.compile()
    model.load_weight(models_path,model_name=model_name)
    (x_train, y_train), (x_test, y_test), (vocab, chunk_tags) = process_data.bert_load_data(train_path, test_path)
    
    predict_tags=[]
    true_tags=[]
    for i in range(len(x_train)):
        predict_text=x_train[i]
        s,length=process_data.process_data(predict_text,vocab)
        raw = model.predict(s)[0][-length:]
        result = [np.argmax(row) for row in raw]
        for j in range(len(result)):
            predict_tags.append(int(result[j]))
            true_tags.append(y_train[i][j])
    # target_names=[0,1,2]
    print(classification_report(true_tags, predict_tags))

if __name__=='__main__':
    model_name='ner2_9_e8'
    # ner_val(model_name)
    ner_prf(model_name)