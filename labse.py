# LaBSE-en-ru
# https://huggingface.co/cointegrated/LaBSE-en-ru

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from labse_predict import IntentRecognizer
from labse_train import train_torch

device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'


def print_input(**kwargs):
    print(kwargs)


def test():
    memory_before = torch.cuda.memory_allocated(0)
    tokenizer = AutoTokenizer.from_pretrained("cointegrated/LaBSE-en-ru")
    model = AutoModelForSequenceClassification.from_pretrained("cointegrated/LaBSE-en-ru", num_labels=2)
    model.to(device)
    memory_after = torch.cuda.memory_allocated(0)
    print((memory_after - memory_before) / 1024 / 1024, "MB")
    sentences = ["Hello World, and trash in end", "Привет Мир?", "Where is HuggingFace based?"]
    encoded_input = tokenizer(sentences, padding=True, truncation=True, max_length=64, return_tensors='pt')
    encoded_input['input_ids'] = encoded_input['input_ids'].to(device)
    encoded_input['token_type_ids'] = encoded_input['token_type_ids'].to(device)
    encoded_input['attention_mask'] = encoded_input['attention_mask'].to(device)
    with torch.no_grad():
        model_output = model(**encoded_input)
        print_input(**encoded_input)
    embeddings = model_output.pooler_output
    print(embeddings)
    # norm_embeddings = torch.nn.functional.normalize(embeddings, dim=0)
    # t = torch.cuda.get_device_properties(0).total_memory
    # r = torch.cuda.memory_reserved(0)


if __name__ == '__main__':
    # test()
    ir = IntentRecognizer(device)
    ir.predict("Что интересного в ИТ?")
    ir.predict(["Я хочу узнать направление 09.03.04.", "Расскажи о IT"])
    print("Цифра чтоб выйти")
    while True:
        in_text = input("-> ")
        if str(in_text).isnumeric():
            break
        out_text = ir.predict(in_text)
        print(f"=> {out_text}")
# Косинусная близость
# Привет
# Bert Topic
# Random Forest
