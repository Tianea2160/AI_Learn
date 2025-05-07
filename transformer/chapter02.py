import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.preprocessing import MinMaxScaler
from transformers import AutoModel
from transformers import DistilBertTokenizer
from umap import UMAP

emotions = load_dataset("emotion")
print(emotions)

emotion_ds = emotions['train']
print(emotion_ds)  # 학습용 데이터 셋
print(len(emotion_ds))  # 데이터 셋 크기
print(emotion_ds.features)  # 데이터 타입
print(emotion_ds.column_names)  # 컬럼 이름
print(emotion_ds[:5])  # 5개만 슬라이스

emotions.set_format(type='pandas')
df = emotions['train'][:]
print(df.head())


def label_int2str(row):
    return emotions['train'].features["label"].int2str(row)


df['label_name'] = df['label'].apply(label_int2str)
print(df.head())

df['label_name'].value_counts(ascending=True).plot.barh()
plt.title('Frequency of Classes')
plt.show()

df["Words Per Tweet"] = df["text"].str.split().apply(len)
df.boxplot("Words Per Tweet", by="label_name", grid=False, showfliers=False, color="black")
plt.suptitle("")
plt.xlabel("")
plt.show()

emotions.reset_format()

## 문자 수준의 토큰화
text = "Tokenizing text is a core task of NLP."
tokenized_text = list(text)
print(tokenized_text)

token2idx = {ch: idx for idx, ch in enumerate(set(tokenized_text))}
print(token2idx)

input_idx = [token2idx[token] for token in tokenized_text]
print(input_idx)

input_idx = torch.tensor(input_idx)
one_hot_encodings = F.one_hot(input_idx, num_classes=len(token2idx))
print(one_hot_encodings.shape)

# 원시 단어 수준의 토큰화
tokenized_text = text.split()
print(tokenized_text)

# 단어 수준의 토큰화
model_ckpt = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(model_ckpt)

encoded_text = tokenizer(text)
print(encoded_text)

tokens = tokenizer.convert_ids_to_tokens(encoded_text.input_ids)
print(tokens)

print(tokenizer.convert_tokens_to_string(tokens))

print(tokenizer.vocab_size)
print(tokenizer.model_max_length)
print(tokenizer.model_input_names)


def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)


print(tokenize(emotions["train"][:2]))

emotions_encode = emotions.map(tokenize, batched=True, batch_size=None)
print(emotions_encode["train"].column_names)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(model_ckpt).to(device)

text = "This is a test"
inputs = tokenizer(text, return_tensors="pt")
print(f"입력된 텐서 크기 : {inputs['input_ids'].size()}")

inputs = {k: v.to(device) for k, v in inputs.items()}
print(inputs)
with torch.no_grad():
    outputs = model(**inputs)
print(outputs)

print(outputs.last_hidden_state.size())
print(outputs.last_hidden_state[:, 0].size())


def extract_hidden_states(batch):
    inputs = {k: v.to(device) for k, v in batch.items() if k in tokenizer.model_input_names}
    with torch.no_grad():
        last_hidden_state = model(**inputs).last_hidden_state
    return {"hidden_state": last_hidden_state[:, 0].cpu().numpy()}


emotions_encode.set_format("torch", columns=["input_ids", "attention_mask", "label"])
emotion_hidden = emotions_encode.map(extract_hidden_states, batched=True)

print(emotion_hidden["train"].column_names)

x_train = np.array(emotion_hidden["train"]["hidden_state"])
x_valid = np.array(emotion_hidden["validation"]["hidden_state"])
y_train = np.array(emotion_hidden["train"]["label"])
y_valid = np.array(emotion_hidden["validation"]["label"])

print(x_train.shape, x_valid.shape)

x_scaled = MinMaxScaler().fit_transform(x_train)
mapper = UMAP(n_components=2, metric='cosine').fit(x_scaled)
df_emb = pd.DataFrame(mapper.embedding_, columns=["X", "Y"])
df_emb["label"] = y_train
print(df_emb.head())
