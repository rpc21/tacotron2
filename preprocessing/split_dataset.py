import pandas as pd
import pickle
import numpy as np
import torch
from sklearn.model_selection import train_test_split

PATH_TO_DATA = '/scratch/speech/datasets/IEMOCAP.pkl'

ENCODING = {
    'fru': torch.tensor([1]),
}

if __name__=='__main__':
    with open(PATH_TO_DATA, 'rb') as f:
        data = pickle.load(f)
    df = pd.DataFrame(list(zip(data['filename'],data['text'],data['label'])),columns=['files','text','label'])
    df = df[df['label'].isin(ENCODING.keys())]
    file_train, file_val, text_train, text_val, label_train, label_val = train_test_split(df['files'], df['text'], df['label'], test_size=0.2, random_state=1234)
    file_train = [x for x in file_train]
    file_val = [x for x in file_val]
    text_train = [x for x in text_train]
    text_val = [x for x in text_val]
    label_train = [ENCODING[x] for x in label_train]
    label_val = [ENCODING[x] for x in label_val]
    train = {
        'filename': file_train,
        'text': text_train,
        'label': label_train
    }

    with open('/scratch/speech/datasets/IEMOCAP_fru_train.pkl','wb') as f:
        pickle.dump(train,f)

    val = {
        'filename': file_val,
        'text': text_val,
        'label': label_val
    }

    with open('/scratch/speech/datasets/IEMOCAP_fru_val.pkl','wb') as f:
        pickle.dump(val,f)

