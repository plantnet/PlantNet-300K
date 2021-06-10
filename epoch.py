import torch
from tqdm import tqdm
from utils import count_correct_top_k, count_correct_average_k
import torch.nn.functional as F
from collections import defaultdict


def train_epoch(model, optimizer, train_loader, criteria, loss_train, train_accuracy, topk_train_accuracy, list_k, n_train, use_gpu):
    model.train()
    loss_epoch_train = 0
    n_correct_train = 0
    n_correct_top_k_train = defaultdict(int)
    epoch_top_k_accuracy_train = {}
    for batch_idx, (batch_x_train, batch_y_train) in enumerate(tqdm(train_loader, desc='train', position=0)):
        if use_gpu:
            batch_x_train, batch_y_train = batch_x_train.cuda(), batch_y_train.cuda()
        optimizer.zero_grad()
        batch_output_train = model(batch_x_train)

        loss_batch_train = criteria(batch_output_train, batch_y_train)
        loss_epoch_train += loss_batch_train.item()
        loss_batch_train.backward()
        optimizer.step()
        with torch.no_grad():
            n_correct_train += torch.sum(torch.eq(batch_y_train, torch.argmax(batch_output_train, dim=-1))).item()
            for k in list_k:
                n_correct_top_k_train[k] += count_correct_top_k(scores=batch_output_train, labels=batch_y_train,
                                                                k=k).item()
    # At the end of epoch compute average of statistics over batches and store them
    with torch.no_grad():
        loss_epoch_train /= batch_idx
        epoch_accuracy_train = n_correct_train / n_train
        for k in list_k:
            epoch_top_k_accuracy_train[k] = n_correct_top_k_train[k] / n_train

        loss_train.append(loss_epoch_train), train_accuracy.append(epoch_accuracy_train), topk_train_accuracy.append(
            epoch_top_k_accuracy_train)

    return loss_epoch_train, epoch_accuracy_train, epoch_top_k_accuracy_train


def val_epoch(model, val_loader, criteria, loss_val, val_accuracy, topk_val_accuracy, averagek_val_accuracy, list_k,
              dataset_attributes, use_gpu):

    model.eval()
    with torch.no_grad():
        loss_epoch_val = 0
        n_correct_val = 0
        n_correct_top_k_val = defaultdict(int)
        epoch_top_k_accuracy_val, epoch_average_k_accuracy_val, lmbda_val = {}, {}, {}
        n_correct_average_k_val = defaultdict(int)

        list_val_proba = []
        list_val_labels = []
        for batch_idx, (batch_x_val, batch_y_val) in enumerate(tqdm(val_loader, desc='val', position=0)):
            if use_gpu:
                batch_x_val, batch_y_val = batch_x_val.cuda(), batch_y_val.cuda()
            batch_output_val = model(batch_x_val)
            batch_proba = F.softmax(batch_output_val)
            list_val_proba.append(batch_proba)
            list_val_labels.append(batch_y_val)

            loss_batch_val = criteria(batch_output_val, batch_y_val)
            loss_epoch_val += loss_batch_val.item()

            n_correct_val += torch.sum(torch.eq(batch_y_val, torch.argmax(batch_output_val, dim=-1))).item()
            for k in list_k:
                n_correct_top_k_val[k] += count_correct_top_k(scores=batch_output_val, labels=batch_y_val, k=k).item()

        val_probas = torch.cat(list_val_proba)
        val_labels = torch.cat(list_val_labels)
        flat_val_probas = torch.flatten(val_probas)
        sorted_probas, _ = torch.sort(flat_val_probas, descending=True)

        for k in list_k:
            lmbda_val[k] = 0.5 * (sorted_probas[dataset_attributes['n_val'] * k - 1] + sorted_probas[dataset_attributes['n_val'] * k])
            n_correct_average_k_val[k] += count_correct_average_k(probas=val_probas, labels=val_labels, lmbda=lmbda_val[k]).item()

        # After seeing val update the statistics over batches and store them
        loss_epoch_val /= batch_idx
        epoch_accuracy_val = n_correct_val / dataset_attributes['n_val']
        for k in list_k:
            epoch_top_k_accuracy_val[k] = n_correct_top_k_val[k] / dataset_attributes['n_val']
            epoch_average_k_accuracy_val[k] = n_correct_average_k_val[k] / dataset_attributes['n_val']

        loss_val.append(loss_epoch_val), val_accuracy.append(epoch_accuracy_val), topk_val_accuracy.append(
            epoch_top_k_accuracy_val), averagek_val_accuracy.append(epoch_average_k_accuracy_val)

    return loss_epoch_val, epoch_accuracy_val, epoch_top_k_accuracy_val, epoch_average_k_accuracy_val, lmbda_val


def test_epoch(model, test_loader, criteria, list_k, lmbda, use_gpu, n_test):

    print()
    model.eval()
    with torch.no_grad():
        loss_epoch_test = 0
        n_correct_test = 0
        epoch_top_k_accuracy_test, epoch_average_k_accuracy_test = {}, {}
        n_correct_top_k_test, n_correct_average_k_test = defaultdict(int), defaultdict(int)
        for batch_idx, (batch_x_test, batch_y_test) in enumerate(tqdm(test_loader, desc='test', position=0)):
            if use_gpu:
                batch_x_test, batch_y_test = batch_x_test.cuda(), batch_y_test.cuda()
            batch_output_test = model(batch_x_test)
            batch_ouput_probra_test = F.softmax(batch_output_test)
            loss_batch_test = criteria(batch_output_test, batch_y_test)
            loss_epoch_test += loss_batch_test.item()

            n_correct_test += torch.sum(torch.eq(batch_y_test, torch.argmax(batch_output_test, dim=-1))).item()
            for k in list_k:
                n_correct_top_k_test[k] += count_correct_top_k(scores=batch_output_test, labels=batch_y_test, k=k).item()
                n_correct_average_k_test[k] += count_correct_average_k(probas=batch_ouput_probra_test, labels=batch_y_test,
                                                                       lmbda=lmbda[k]).item()

        # After seeing test test update the statistics over batches and store them
        loss_epoch_test /= batch_idx
        epoch_accuracy_test = n_correct_test / n_test
        for k in list_k:
            epoch_top_k_accuracy_test[k] = n_correct_top_k_test[k] / n_test
            epoch_average_k_accuracy_test[k] = n_correct_average_k_test[k] / n_test

    return loss_epoch_test, epoch_accuracy_test, epoch_top_k_accuracy_test, epoch_average_k_accuracy_test
