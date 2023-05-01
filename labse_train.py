import datetime
import gc
import random
import time
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler, \
    WeightedRandomSampler
from transformers import TrainingArguments, Trainer, AutoModelForSequenceClassification, BatchEncoding, \
    get_linear_schedule_with_warmup, AutoTokenizer, BertForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput

device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained("cointegrated/LaBSE-en-ru")


def train():
    df = pd.read_csv('dataset.csv', sep=';', comment='#')
    sentences = df.iloc[:, 1]
    answers = df.iloc[:, 0]

    from datasets import Dataset
    raw_dataset = Dataset.from_csv('dataset.csv', sep=';', comment='#')

    tokenized_dataset = raw_dataset.map(tokenization, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "label"])
    print(tokenized_dataset.format['type'])

    training_args = TrainingArguments(
        output_dir='./results',
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        fp16=True,
        fp16_opt_level='O2',
        no_cuda=False
    )
    model = AutoModelForSequenceClassification.from_pretrained("cointegrated/LaBSE-en-ru", num_labels=2)
    model.to(device)
    # Создание экземпляра класса Trainer для обучения модели
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,  # На чем дообучаем
    )
    trainer.train()


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """
    # Round to the nearest second.
    elapsed_rounded = int(round(elapsed))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


class TextDataset(Dataset):
    def __init__(self, file_path: str, transform: Callable):
        df = pd.read_csv(file_path, sep=';', comment='#')
        self.item_labels = df.iloc[:, 0]
        self.item_texts = df.iloc[:, 1]
        self._item_tokens = [None for _ in range(len(self.item_labels))]
        self.transform = transform

    def __len__(self):
        return len(self.item_labels)

    def __getitem__(self, item):
        if self._item_tokens[item] is None:
            item_token = self.transform({"text": self.item_texts[item]})
            self._item_tokens[item] = item_token
        return self._item_tokens[item], self.item_labels[item]


def tokenization(example):
    tokenized = tokenizer(example["text"], padding=True, truncation=True, max_length=64, return_tensors='pt')
    return tokenized


def print_layers(model):
    params = list(model.named_parameters())

    print('The BERT model has {:} different named parameters.\n'.format(len(params)))

    print('==== Embedding Layer ====\n')

    for p in params[0:5]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    print('\n==== First Transformer ====\n')

    for p in params[5:21]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    print('\n==== Output Layer ====\n')

    for p in params[-4:]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
    del params, p
    gc.collect()


def train_torch(epochs=2):
    # Set the seed value all over the place to make this reproducible.
    seed_val = 2023

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    text_dataset = TextDataset('dataset.csv', tokenization)

    model: BertForSequenceClassification = AutoModelForSequenceClassification.from_pretrained(
        "cointegrated/LaBSE-en-ru", num_labels=len(set(text_dataset.item_labels)))
    model.to(device)

    tokens: BatchEncoding = tokenization({"text": list(text_dataset.item_texts)})
    input_ids = tokens.data['input_ids'].to(device)
    attention_masks = tokens.data['attention_mask'].to(device)
    labels = torch.tensor(text_dataset.item_labels)

    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(input_ids, attention_masks, labels)

    # Create a 90-10 train-validation split.

    # Calculate the number of samples to include in each set.
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    # Divide the dataset by randomly selecting samples.
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_labels = train_dataset[:][2]
    print('{:>5,} training samples'.format(train_size))
    print('{:>5,} validation samples'.format(val_size))
    # print(train_dataset[0])
    for i in range(len(train_dataset)):
        a: BatchEncoding = train_dataset[i][0]
        # print(a.data)
        if i >= len(train_dataset) - 1:
            continue
        b: BatchEncoding = train_dataset[i][0]
        assert a.shape == b.shape
    val_dataset = train_dataset
    # The DataLoader needs to know our batch size for training, so we specify it
    # here. For fine-tuning BERT on a specific task, the authors recommend a batch
    # size of 16 or 32.
    batch_size = 32

    # Sampler for imbalanced datasets
    class_sample_count = np.array(
        [len(np.where(train_labels == t)[0]) for t in np.unique(train_labels)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in train_labels])
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))

    # Create the DataLoaders for our training and validation sets.
    # We'll take training samples in random order.
    train_dataloader = DataLoader(
        train_dataset,  # The training samples.
        sampler=sampler,  # Select batches randomly
        batch_size=batch_size  # Trains with this batch size.
    )

    # For validation the order doesn't matter, so we'll just read them sequentially.
    validation_dataloader = DataLoader(
        val_dataset,  # The validation samples.
        sampler=SequentialSampler(val_dataset),  # Pull out batches sequentially.
        batch_size=batch_size  # Evaluate with this batch size.
    )
    # Note: AdamW is a class from the huggingface library (as opposed to pytorch)
    # I believe the 'W' stands for 'Weight Decay fix'
    optimizer = AdamW(model.parameters(),
                      lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                      )
    # Number of training epochs. The BERT authors recommend between 2 and 4.
    # We chose to run for 4, but we'll see later that this may be over-fitting the
    # training data.
    # epochs = 4

    # Total number of training steps is [number of batches] x [number of epochs].
    # (Note that this is not the same as the number of training samples).
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)
    # Get all the model's parameters as a list of tuples.
    print_layers(model)

    ###

    # This training code is based on the `run_glue.py` script here:
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

    # We'll store a number of quantities such as training and validation loss,
    # validation accuracy, and timings.
    training_stats = []

    # Measure the total training time for the whole run.
    total_t0 = time.time()

    best_avg_val_accuracy = 0

    # For each epoch...
    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0

        # Put the model into training mode. Don't be mislead--the call to
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: labels
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            gc.collect()
            torch.cuda.empty_cache()

            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because
            # accumulating the gradients is "convenient while training RNNs".
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()

            # Perform a forward pass (evaluate the model on this training batch).
            # The documentation for this `model` function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # It returns different numbers of parameters depending on what arguments
            # arge given and what flags are set. For our usage here, it returns
            # the loss (because we provided labels) and the "logits"--the model
            # outputs prior to activation.
            model_output: SequenceClassifierOutput = model(b_input_ids,
                                                           token_type_ids=None,
                                                           attention_mask=b_input_mask,
                                                           labels=b_labels)
            loss, logits = model_output.loss, model_output.logits

            # Accumulate the training loss over all the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value
            # from the tensor.
            total_train_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.

            # print(torch.cuda.memory_summary(device=None, abbreviated=False))
            optimizer.step()
            # print(torch.cuda.memory_summary(device=None, abbreviated=False))

            # Update the learning rate.
            scheduler.step()
            gc.collect()
            torch.cuda.empty_cache()

        # Calculate the average loss over all the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)

        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(training_time))

        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        print("")
        print("Running Validation...")

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()

        # Tracking variables
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:
            # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using
            # the `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: labels
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            gc.collect()
            torch.cuda.empty_cache()

            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():
                # Forward pass, calculate logit predictions.
                # token_type_ids is the same as the "segment ids", which
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                # The documentation for this `model` function is here:
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                # Get the "logits" output by the model. The "logits" are the output
                # values prior to applying an activation function like the softmax.
                model_output = model(b_input_ids,
                                     token_type_ids=None,
                                     attention_mask=b_input_mask,
                                     labels=b_labels)
                loss, logits = model_output.loss, model_output.logits

            # Accumulate the validation loss.
            total_eval_loss += loss.item()

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.
            total_eval_accuracy += flat_accuracy(logits, label_ids)

        # Report the final accuracy for this validation run.
        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

        # Calculate the average loss over all the batches.
        avg_val_loss = total_eval_loss / len(validation_dataloader)

        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )
        if avg_val_accuracy > best_avg_val_accuracy:
            best_avg_val_accuracy = avg_val_accuracy
            torch.save(model, 'cool_model.pt')
            print("")
            print("Saving best model.")

    print("")
    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))


def main():
    train_torch(5)


if __name__ == '__main__':
    main()
