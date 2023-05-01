# Распознавание интентов
## Требования
* \>= Python 3.10
* CUDA
* \>= 8GB VRAM
* \>= 16GB RAM for Docker

## Как запустить?
### Зависимости
> pip install -r requirements.txt
### Обучение
> py labse_train.py
### Запуск демо
> py labse.py

# API
Есть `labse_predict.py`, а в метод `predict(msg) -> int`, который принимает на вход текст и возвращает номер интента.