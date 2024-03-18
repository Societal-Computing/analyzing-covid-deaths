import time
import datetime

import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm


# Some helper functions for training
#
# For DATA
def get_ids_and_attn_masks(tokenizer, sentences, labels=None, max_len=128):
    if not isinstance(sentences, list):
        sentences = list(sentences)

    input_ids = []
    attention_masks = []
    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
            sent,
            add_special_tokens=True,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict["input_ids"])

        # And its attention mask
        attention_masks.append(encoded_dict["attention_mask"])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    if labels is not None:
        labels = torch.tensor(list(labels))

    return input_ids, attention_masks, labels


# For model training
#
# To show the time taken for training
def format_time(elapsed):
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


# To calculate the f1 score
def calculate_f1_score(preds, labels):
    preds = F.softmax(preds, dim=1)

    # Move logits and labels to CPU
    preds = preds.detach().cpu().numpy()
    labels = labels.to("cpu").numpy()

    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = np.argmax(labels, axis=1).flatten()

    f1 = f1_score(pred_flat, labels_flat, average="weighted")

    return f1, list(pred_flat), list(labels_flat)


# For training
def train(
    model, train_dl, val_dl, num_epochs=3, n_classes=2, learning_rate=2e-5, DEVICE="cpu"
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    criterion = nn.CrossEntropyLoss()

    # Measure the total training time for the whole run.
    total_t0 = time.time()

    # For each epoch...
    for epoch_i in range(0, num_epochs):
        print("")
        print("======== Epoch {:} / {:} ========".format(epoch_i + 1, num_epochs))
        print("Training...")

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0

        model.train()

        for batch in tqdm(train_dl):
            b_input_ids = batch[0].to(DEVICE)
            b_input_mask = batch[1].to(DEVICE)
            b_labels = batch[2].to(DEVICE)

            b_labels = (
                F.one_hot(b_labels, num_classes=n_classes)
                .type(torch.FloatTensor)
                .to(DEVICE)
            )

            model.zero_grad()

            logits = model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
            )
            loss = criterion(logits, b_labels)

            total_train_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dl)

        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))

        if val_dl is not None:
            # if validation set is provided, do the validation

            print("")
            print("Running Validation...")

            t0 = time.time()

            # Put the model in evaluation mode
            model.eval()

            # Tracking variables
            total_eval_f1 = 0
            total_eval_loss = 0

            # Evaluate data for one epoch
            for batch in tqdm(val_dl):
                b_input_ids = batch[0].to(DEVICE)
                b_input_mask = batch[1].to(DEVICE)
                b_labels = batch[2].to(DEVICE)

                b_labels = (
                    F.one_hot(b_labels, num_classes=n_classes)
                    .type(torch.FloatTensor)
                    .to(DEVICE)
                )

                with torch.no_grad():
                    logits = model(
                        b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask,
                    )

                    loss = criterion(logits, b_labels)

                # Accumulate the validation loss.
                total_eval_loss += loss.item()

                # Calculate the f1-score for this batch of sentences, and
                f1, preds, labs = calculate_f1_score(logits, b_labels)

                total_eval_f1 += f1

            # Report the final accuracy for this validation run.
            avg_f1 = total_eval_f1 / len(val_dl)

            print("  F1: {0:.2f}".format(avg_f1))

            # Calculate the average loss over all of the batches.
            avg_val_loss = total_eval_loss / len(val_dl)

            # Measure how long the validation run took.
            validation_time = format_time(time.time() - t0)

            print("  Validation Loss: {0:.2f}".format(avg_val_loss))
            print("  Validation took: {:}".format(validation_time))

    print("")
    print("Training complete!")

    print(
        "Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0))
    )

    # Save the model checkpoint
    torch.save(model.state_dict(), "out/model/model.bin")

    return model


def evaluate_test(model, test_dl, n_classes=2, DEVICE="cpu"):
    print("Running Test...")

    criterion = nn.CrossEntropyLoss()

    t0 = time.time()

    # Put the model in evaluation mode
    model.eval()

    # Tracking variables
    f1_scores = []

    total_eval_loss = 0

    predictions = []
    labels = []

    # Evaluate data for one epoch
    for batch in tqdm(test_dl):
        b_input_ids = batch[0].to(DEVICE)
        b_input_mask = batch[1].to(DEVICE)
        b_labels = batch[2].to(DEVICE)

        b_labels = (
            F.one_hot(b_labels, num_classes=n_classes)
            .type(torch.FloatTensor)
            .to(DEVICE)
        )

        with torch.no_grad():
            logits = model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
            )
            loss = criterion(logits, b_labels)

        # Accumulate the validation loss.
        total_eval_loss += loss.item()

        f1, preds, labs = calculate_f1_score(logits, b_labels)
        f1_scores.append(f1)

        predictions.extend(preds)
        labels.extend(labs)

    # Report the final accuracy for this validation run.
    avg_f1 = np.array(f1_scores).mean()
    f1_std = np.array(f1_scores).std()

    print("  F1: {0:.2f}".format(avg_f1))
    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / len(test_dl)

    # Measure how long the validation run took.
    validation_time = format_time(time.time() - t0)

    print("  Test Loss: {0:.2f}".format(avg_val_loss))
    print("  Test took: {:}".format(validation_time))

    confusion = confusion_matrix(labels, predictions)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion, display_labels=["no_label", "yes_label"]
    )

    disp.plot()
    disp.figure_.savefig("out/plots/test_evaluation_confusion.png")

    return predictions, labels, avg_f1, f1_std


def make_predictions(model, dl, DEVICE="cpu"):
    print("Running Predictions...")
    model.eval()

    predictions = []

    for batch in tqdm(dl):
        b_input_ids = batch[0].to(DEVICE)
        b_input_mask = batch[1].to(DEVICE)

        with torch.no_grad():
            logits = model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
            )

        # Move logits and labels to CPU
        preds = F.softmax(logits, dim=1)
        preds = preds.detach().cpu().numpy()

        pred_flat = np.argmax(preds, axis=1).flatten()
        predictions.extend(pred_flat)

    return predictions
