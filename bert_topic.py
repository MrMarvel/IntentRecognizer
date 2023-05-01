# BERTopic
# https://github.com/MaartenGr/BERTopic

import pandas as pd
import torch
from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups


def main():
    print("START")
    a = torch.cuda.memory_allocated(0)
    device = torch.cuda.device(0) if torch.cuda.is_available() else 'cpu'
    docs: list[str] = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))['data'][:100]
    docs = [msg.replace('\n','') for msg in docs]
    # df = pd.read_table('dataset.csv')
    # docs = list(df.iloc[:, 0])
    print(docs)
    topic_model = BERTopic(language='russian')
    topics, probs = topic_model.fit_transform(docs)
    print(pd.DataFrame(topics))
    print(topics, probs)
    print(pd.DataFrame({"topic": topics, "document": docs}))
    a_new = torch.cuda.memory_allocated(0)
    print((a_new - a) / 1024 / 1024, "MB")


if __name__ == '__main__':
    main()
