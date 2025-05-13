import pandas as pd
from transformers import pipeline

text = """Dear Amazon, last week I ordered an Optimus Prime action figure \
from your online store in Germany. Unfortunately, when I opened the package, \
I discovered to my horror that I had been sent an action figure of Megatron \
instead! As a lifelong enemy of the Decepticons, I hope you can understand my \
dilemma. To resolve the issue, I demand an exchange of Megatron for the \
Optimus Prime figure I ordered. Enclosed are copies of my records concerning \
this purchase. I expect to hear from you soon. Sincerely, Bumblebee."""

# 감정
classifier = pipeline("text-classification")
outputs = classifier(text)
print(pd.DataFrame(outputs))

# 단어 선택
ner_tagger = pipeline("ner", aggregation_strategy="simple")
outputs = ner_tagger(text)
print(pd.DataFrame(outputs))

# 질문
reader = pipeline("question-answering")
question = "What does the customer want?"

input_dict = {
    "question": question,
    "context": text
}
outputs = reader(input_dict)
print(pd.DataFrame([outputs]))

# 요약
summarizer = pipeline("summarization")
outputs = summarizer(text, max_length=45, clean_up_tokenization_spaces=True)
print(outputs[0]['summary_text'])

# 번역
translator = pipeline("translation_en_to_de", model="Helsinki-NLP/opus-mt-en-de")
outputs = translator(text, clean_up_tokenization_spaces=True, min_length=100)
print(outputs[0]['translation_text'])

# 텍스트 생성
generator = pipeline("text-generation")
response = "Dear Bumblebee, I am sorry to hear that your order was mixed up."
prompt = text + "\n\nCustomer service response:\n" + response
outputs = generator(prompt, max_length=200)
print(outputs[0]['generated_text'])
