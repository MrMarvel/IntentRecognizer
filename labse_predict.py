import numpy as np
import torch
from transformers import AutoTokenizer


class IntentRecognizer:
    def __init__(self, device):
        self._device = device
        self._tokenizer = AutoTokenizer.from_pretrained("cointegrated/LaBSE-en-ru")
        self._model = torch.load('cool_model.pt')
        self._model.to(device)

    def predict(self, msg):
        memory_before = torch.cuda.memory_allocated(0)
        memory_after = torch.cuda.memory_allocated(0)
        print((memory_after - memory_before) / 1024 / 1024, "MB")
        sentences = msg
        encoded_input = self._tokenizer(sentences, padding=True, truncation=True, max_length=64, return_tensors='pt')
        encoded_input['input_ids'] = encoded_input['input_ids'].to(self._device)
        encoded_input['token_type_ids'] = encoded_input['token_type_ids'].to(self._device)
        encoded_input['attention_mask'] = encoded_input['attention_mask'].to(self._device)
        with torch.no_grad():
            model_output = self._model(**encoded_input)
        logits = model_output.logits.cpu()
        predictions = torch.nn.functional.softmax(logits, dim=-1)
        print(predictions)
        answer = np.argmax(predictions, axis=-1)
        print(answer)
        return answer
        # norm_embeddings = torch.nn.functional.normalize(embeddings, dim=0)
        # t = torch.cuda.get_device_properties(0).total_memory
        # r = torch.cuda.memory_reserved(0)
