import os.path
import random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import dialogues_data_loader
from transformer_model import BertTextSpansModel
from dataset import TextSpansDatasetForTransformers

import numpy as np
import re
import json
import config
from dataset import text_spans_classes

import matplotlib.pyplot as plt

def load_dialogues_texts(data_filepath):
    print("Loading dialogues data...")
    texts = []
    classes = []
    with open(data_filepath, encoding='utf8') as file:
        # skip the first line --> header
        all_lines = file.readlines()[1:]

    # shuffle all lines
    random.shuffle(all_lines)
    for line in all_lines:
        line_elements = line.split("\t")
        text = line_elements[-1]
        label = line_elements[0]
        texts.append(text.strip())
        classes.append(label)

    assert len(classes) == len(texts)
    print("Loaded ", len(texts), " texts")
    print("Loaded ", len(classes), " classes")

    print("Spliting into train and test parts")
    # 0.7 training data, 0.3 test data
    border_index = int(len(texts) * 0.7)
    train_texts, train_classes = texts[:border_index], classes[:border_index]
    test_texts, test_classes = texts[border_index:], classes[border_index:]

    return train_texts, train_classes, test_texts, test_classes

def bert_experiment_text_spans(train_data, test_data):
    modelid = "bert-large-cased"
    number_of_classes = len(text_spans_classes)

    model = BertTextSpansModel(modelid, device="cuda", out_dim=number_of_classes)
    # print(model)
    print(model.get_num_of_learnable_params(), " trainable params")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.00001)

    modelpath = "text_spans_models/bert-large-cased/model_best_val_acc.cp"
    if os.path.exists(modelpath):
        model.load_learnable_params(modelpath)
        model.criterion = criterion
        test_dataloader = DataLoader(test_data, batch_size=16, shuffle=False)
        test_results = model.loop_test(test_dataloader)
        return test_results
    else:
        train_results, val_results = model.train_model(
            train_dataset=train_data,
            val_dataset=test_data,
            epochs=5,
            criterion=criterion,
            optimizer=optimizer,
            batch=16
        )

        print("Train results: ", train_results)
        print("Val results: ", val_results)
        return val_results["acc"]


def get_word_index_according_to_string_index(string, index):
    position = 0
    word_position = 0
    for i, word in enumerate(string):
        position += (1 + len(word))
        if i >= index:
            break
        if word == " ":
            word_position += 1
    return word_position

def get_text_span_indices(text_span, utterance_text):
    # print("Utterance: ", utterance_text)
    # print("Text span: ", text_span)
    start_word_index = 0
    end_word_index = len(utterance_text.split()) - 1

    for match in re.finditer(text_span, utterance_text):
        # print(match.start(), match.end())
        start_word_index = get_word_index_according_to_string_index(utterance_text, match.start())
        # print("START WORD INDEX = ", start_word_index)
        end_word_index = get_word_index_according_to_string_index(utterance_text, match.end())
        # print("END WORD INDEX = ", end_word_index)
        # just first appearing
        break
    return start_word_index, end_word_index

def create_text_span_class(text_span, utterance_text):
    if text_span == utterance_text:
        return "whole_utterance"
    else:
        pattern = r'[.,!?;]'
        text_span = re.sub(pattern, "", text_span).strip()
        utt_parts = re.split(pattern, utterance_text)
        updated_utt_parts = []
        for utt_part in utt_parts:
            # if number of words is above zero
            if len(utt_part.split()) > 0:
                updated_utt_parts.append(utt_part)
        utt_parts = updated_utt_parts
        if len(utt_parts) > 0:
            first_part = utt_parts[0].strip()
            middle_part = utt_parts[len(utt_parts) // 2].strip()
            last_part = utt_parts[-1].strip()
            if text_span == first_part:
                return "first_part"
            elif text_span == last_part:
                return "last_part"
            elif text_span == middle_part:
                return "middle_part"
            else:
                return "other"
        else:
            return "other"

def create_data(data, dataset_type_list):
    all_texts, all_classes = [], []
    for item in data:
        utt_texts = []
        texts = []
        classes = []
        conversation = item["conversation"]
        for utt in conversation:
            # filter out the utterances/conversation that don't belong to the dataset we creating
            conv_id = "dia"+str(item["conversation_ID"])
            utt_id = "utt"+str(utt["utterance_ID"])
            identifier = conv_id+utt_id+".mp4"
            if dataset_type_list is not None and identifier not in dataset_type_list:
                continue
            utt_texts.append(utt["text"])
        for emotion_pair in item["emotion-cause_pairs"]:
            if len(utt_texts) == 0:
                break
            text_span_item = emotion_pair[1] # the second item is text span
            text_span = text_span_item.split("_")[1]
            utterance_id = text_span_item.split("_")[0]
            utterance_text = utt_texts[int(utterance_id) - 1]
            label = create_text_span_class(text_span, utterance_text)
            classes.append(label)
            texts.append(utterance_text)

        all_texts.extend(texts)
        all_classes.extend(classes)

    return all_texts, all_classes

def predict_text_spans():
    tokenizerid = "bert-base-cased"
    predicted_text_spans, predicted_text_span_labels, predicted_indices = [], [], []

    # load all data
    # Opening JSON file
    f = open(config.eval_data_path)
    # returns JSON object as
    # a dictionary
    data = json.load(f)
    # Closing file
    f.close()

    utt_texts = []
    utt_classes = []
    for item in data:
        conversation = item["conversation"]
        for utt in conversation:
            # filter out the utterances/conversation that don't belong to the dataset we creating
            conv_id = "dia" + str(item["conversation_ID"])
            utt_id = "utt" + str(utt["utterance_ID"])
            identifier = conv_id + utt_id + ".mp4"
            utt_texts.append(utt["text"])
            utt_classes.append("whole_utterance")
    #print(utt_texts)
    print(len(utt_texts), "utterances")

    assert len(utt_texts) == len(utt_classes)

    eval_data = TextSpansDatasetForTransformers(utt_texts, utt_classes, tokenizerid, input_length=100)
    modelid = "bert-large-cased"
    number_of_classes = len(text_spans_classes)

    model = BertTextSpansModel(modelid, device="cuda", out_dim=number_of_classes)
    # print(model)
    print(model.get_num_of_learnable_params(), " trainable params")
    criterion = nn.CrossEntropyLoss()

    modelpath = "text_spans_models/bert-large-cased/model_best_val_acc.cp"
    if os.path.exists(modelpath):
        model.load_learnable_params(modelpath)
        model.criterion = criterion
        eval_dataloader = DataLoader(eval_data, batch_size=1, shuffle=False)
        predicted_classes = model.predict_classes(eval_dataloader)
        print(predicted_classes)
        for i in range(0, len(utt_texts)):
            label = predicted_classes[i]
            text_span_class = text_spans_classes[label]
            predicted_text_span_labels.append(text_span_class)
            if text_span_class == "whole_utterance":
                text_span = utt_texts[i]
                predicted_indices.append("0_" + str(len(utt_texts[i].split()) - 1))
            elif text_span_class == "first_part":
                pattern = r'[.,!?;]'
                utt_parts = re.split(pattern, utt_texts[i])
                if len(utt_parts) > 0:
                    first_part = utt_parts[0].strip()
                    text_span = first_part
                    start_index, end_index = get_text_span_indices(text_span, utt_texts[i])
                    predicted_indices.append(str(start_index)+"_"+str(end_index))
                else:
                    text_span = utt_texts[i]
                    start_index, end_index = get_text_span_indices(text_span, utt_texts[i])
                    predicted_indices.append(str(start_index) + "_" + str(end_index))
            elif text_span_class == "middle_part":
                pattern = r'[.,!?;]'
                utt_parts = re.split(pattern, utt_texts[i])
                if len(utt_parts) > 0:
                    middle_part = utt_parts[len(utt_parts) // 2].strip()
                    text_span = middle_part
                    start_index, end_index = get_text_span_indices(text_span, utt_texts[i])
                    predicted_indices.append(str(start_index) + "_" + str(end_index))
                else:
                    text_span = utt_texts[i]
                    start_index, end_index = get_text_span_indices(text_span, utt_texts[i])
                    predicted_indices.append(str(start_index) + "_" + str(end_index))
            elif text_span_class == "last_part":
                pattern = r'[.,!?;]'
                utt_parts = re.split(pattern, utt_texts[i])
                if len(utt_parts) > 0:
                    if utt_parts[-1].strip() != "":
                        last_part = utt_parts[-1].strip()
                    else:
                        last_part = utt_parts[-2].strip()
                    text_span = last_part
                    start_index, end_index = get_text_span_indices(text_span, utt_texts[i])
                    predicted_indices.append(str(start_index) + "_" + str(end_index))
                else:
                    text_span = utt_texts[i]
                    start_index, end_index = get_text_span_indices(text_span, utt_texts[i])
                    predicted_indices.append(str(start_index) + "_" + str(end_index))
            else:
                # if other --> use whole_sentence
                text_span = utt_texts[i]
                predicted_indices.append("0_" + str(len(utt_texts[i].split()) - 1))

            predicted_text_spans.append(text_span)

    assert len(utt_texts) == len(predicted_text_spans) == len(predicted_text_span_labels) == len(predicted_indices)
    dataframe = pd.DataFrame(
        {
            'Utterance': utt_texts,
            'PredictedTextSpan': predicted_text_spans,
            'PredictedTextSpanLabel': predicted_text_span_labels,
            'PredictedIndices': predicted_indices
        }
    )
    print(dataframe.head())
    dataframe.to_csv("predicted_text_spans_eval_data.tsv", sep="\t")

if __name__ == '__main__':
    only_prediction = True
    if only_prediction:
        predict_text_spans()
    else:
        tokenizerid = "bert-base-cased"

        # load train, test, dev splits
        with open(config.train_test_dev_splits_path + "train.txt") as fw:
            train_diags_list = fw.read().splitlines()
        with open(config.train_test_dev_splits_path + "dev.txt") as fw:
            dev_diags_list = fw.read().splitlines()
        with open(config.train_test_dev_splits_path + "test.txt") as fw:
            test_diags_list = fw.read().splitlines()

        # load all data
        # Opening JSON file
        f = open(config.train_data_text_spans_path)

        # returns JSON object as
        # a dictionary
        data = json.load(f)

        # Closing file
        f.close()

        # labels = 'Frogs', 'Hogs', 'Dogs', 'Logs', "nabs"
        # sizes = [35, 0, 0, 0, 65]
        #
        # fig, ax = plt.subplots()
        # ax.pie(sizes, labels=labels, autopct='%1.1f%%')
        # plt.show()
        # print(labels)
        # print(sizes)

        train_texts, train_classes = create_data(data, dataset_type_list=train_diags_list)
        test_texts, test_classes = create_data(data, dataset_type_list=test_diags_list)


        assert len(train_texts) == len(train_classes)
        assert len(test_texts) == len(test_classes)

        labels = "whole_utterance", "first_part", "last_part", "middle_part", "other"


        train_data = TextSpansDatasetForTransformers(train_texts, train_classes, tokenizerid, input_length=100)
        test_data = TextSpansDatasetForTransformers(test_texts, test_classes, tokenizerid, input_length=100)

        accuracy = bert_experiment_text_spans(train_data, test_data)
        print(f"accuracy: {accuracy}")

