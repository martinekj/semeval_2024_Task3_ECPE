import json
import os.path
from types import SimpleNamespace
import pandas as pd
import copy
import numpy as np
from tqdm import tqdm
import config
from rich import print
from evaluate import predict
from utilz import config as cfg
from dataset import DialogDataset, load_audio_video_vecs
from model import DialogModel
from utils import load_dialogues_json_data, create_json_representation_file, utterances_to_dialogues, \
    dialogues_to_utterances
from utils import create_semeval_format
from dialogues_data_loader import prepare_eval_data


def predict_dlgs(model, ds, dlgs, edit_emotion=True, edit_links=True, filter_neutral=True, t=0.0):
    """
    when combining models, emotion needs to be edited first and links last

    :param model: model for prediction
    :param ds: ds to prepare data
    :param dlgs:    loaded dlgs in Jirka's format
    :param edit_emotion:    if emotions of dlgs should be edited by model's predictions
    :param edit_links:      if links of dlgs should be edited by model's predictions
    :param filter_neutral:  if links with neutral emotion should be filtered
    :param t:               threshold for link predictions
    :return:        edited dlgs
    """
    assert edit_emotion or edit_links, "The function is doing nothing useful with: edit_emotion=False, edit_links=False"

    print(f"predicting edit_emotion={edit_emotion}, edit_links={edit_links}")

    model.train(False)

    for k, dlg in tqdm(dlgs.items(), desc="predicting: "):
        cpred, lpred = predict(ds, dlg, model, t)

        # clear links first of all
        for utt_i, utt in enumerate(dlg):
            utt.emotion_cause_links = []

        if edit_emotion:
            if 0 in cpred[:len(dlg)]:
                raise Exception(f"{config.emotion_categories[0]} emotion can not be predicted in final eval")

            for utt_i, utt in enumerate(dlg):
                utt.utterance_emotion = config.emotion_categories[cpred[utt_i]]  # edit emotion

        if edit_links:
            for srci, trgi in np.argwhere(lpred[:len(dlg), :len(dlg)]):
                if filter_neutral and dlg[trgi].utterance_emotion == "neutral": # neutral emotions are not links
                    continue

                dlg[srci].emotion_cause_links.append({
                    'emotion': dlg[trgi].utterance_emotion,  # target utt emotion
                    'source_id': dlg[srci].utterance_id,
                    'target_id': dlg[trgi].utterance_id
                })

    return dlgs


def load_dlgs(json_file):
    utts = load_dialogues_json_data(json_file)
    dlgs = utterances_to_dialogues(utts)
    return dlgs


def dlgs_to_jirka_json(dlgs, json_file):
    utts = dialogues_to_utterances(dlgs)
    create_json_representation_file(utts, json_file)


def load_model_ds(wfile):
    print(f"Loading {wfile}")
    modeldir = os.path.split(wfile)[0]

    with open(modeldir + "/run_config.json") as f:
        args = json.load(f)["args"]
    args = SimpleNamespace(**args)

    if args.speakers == "all":
        config.speakers = config.all_speakers
    elif args.speakers == "main":
        config.speakers = config.main_speakers
    else:
        raise NotImplementedError("Only 'all' or 'main' options available for speakers")

    load_audio_video_vecs(audiofile="audio_vectors_eval_mfcc2k.tsv", videofile="video_vectors_eval.csv")

    model = DialogModel(
        model=args.modelid,
        speaker_embeds=not args.speaker_off,
        utterance_embeds=not args.utt_off,
        utterance_position_embeds=not args.utt_pos_off,
        enc_hidden_layers=args.enc_hidden_layers,
        max_utts=args.max_utts,
        sqrtatt=args.sqrtatt,
        aggregation=args.aggreg,
        fusion_layers=args.fusion_layers,
        fusion_size=args.fusion_size,
        audio=not args.audio_off,
        video=not args.video_off,
        fusion_aggreg=args.fusion_aggreg,
        proj_lrelu=args.fusion_proj_lrelu
    )
    model.load_learnable_params(wfile)

    ds = DialogDataset(None, tokenizer=args.modelid, input_length=args.inputlen_ds)
    ds.max_n_utt = args.max_utts

    return model, ds


def get_predicted_dlgs(dlgs, model_weights_emotion_path, model_weights_links_path=None, filter_neutral=True):

    if model_weights_links_path is None:    # both
        model, ds = load_model_ds(model_weights_emotion_path)
        return predict_dlgs(model, ds, dlgs, filter_neutral=filter_neutral)
    else:
        model, ds = load_model_ds(model_weights_emotion_path)
        dlgs = predict_dlgs(model, ds, dlgs, edit_links=False, filter_neutral=filter_neutral)

        model, ds = load_model_ds(model_weights_links_path)
        dlgs = predict_dlgs(model, ds, dlgs, edit_emotion=False, filter_neutral=filter_neutral)
        return dlgs


if __name__ == '__main__':
    tmp_folder = cfg.paths.data + "/tmp"
    os.makedirs(tmp_folder, exist_ok=True)
    converted_json_format_filepath = tmp_folder + "/input_our_format.json"
    # converting format into our desired
    prepare_eval_data(converted_json_format_filepath)

    gt_json = converted_json_format_filepath

    output_json = os.path.join(tmp_folder, "result_our_format.json")
    output_semeval_json = "Subtask_2_pred.json"

    # loading model weights
    emotion_weights = cfg.paths.bin + "/TextEmoRoberta-roberta_c1.0_l0.0_warm2/model_ep_39.sd"
    links_weights = cfg.paths.bin + "/minFusion_c0.0005_l1.0_warm2/model_ep_38.sd"

    dlgs = load_dlgs(gt_json)
    dlgs = get_predicted_dlgs(dlgs, model_weights_emotion_path=emotion_weights, model_weights_links_path=links_weights)
    dlgs_to_jirka_json(dlgs, output_json)
    create_semeval_format(output_json, output_semeval_json)

    print(f"FINISHED: result saved to '{output_semeval_json}'")

    ##########################################################################################
    ######          SUBTASK 1 prediction of text spans with bert-large-cased            ######
    ##########################################################################################
    ##########################################################################################
    text_spans_dataframe = pd.read_csv(cfg.paths.data + "/predicted_text_spans_test_data.tsv", sep="\t")

    # load all data
    # Opening JSON file
    f = open("Subtask_2_pred.json")

    # returns JSON object as
    # a dictionary
    data = json.load(f)

    # new_data = data.copy()
    new_data = copy.deepcopy(data)

    # print(new_data)
    i = 0
    first_conv_id = 1375
    for item in data:
        texts, emotions, utt_ids, speakers = [], [], [], []
        conversation_id = item["conversation_ID"]
        for utt in item["conversation"]:
            text = utt["text"]
            emotion = utt["emotion"]
            utt_id = utt["utterance_ID"]
            speaker = utt["speaker"]
            # texts.append(re.sub(r'[^A-Za-z0-9 ]+', '', text))
            texts.append(text)
            emotions.append(emotion)
            utt_ids.append(utt_id)
            speakers.append(speaker)

        # print(conversation_id)
        new_data[int(conversation_id) - first_conv_id]["emotion-cause_pairs"] = []
        for emotion_cause_pair in item["emotion-cause_pairs"]:
            source_utt_index = int(emotion_cause_pair[1].split("_")[0]) - 1
            source_utt = texts[source_utt_index]
            first_part = emotion_cause_pair[0]
            second_part = emotion_cause_pair[1]
            tmp_list = []
            tmp_list.append(first_part)

            j = 0
            pred_indices = None
            # find correct utterance --> pred indices
            for utterance in text_spans_dataframe.Utterance:
                if utterance == "nan":
                    break
                if utterance == source_utt:
                    pred_indices = text_spans_dataframe.PredictedIndices[j]
                    break
                j += 1

            if pred_indices is not None and pred_indices != "0_0":
                tmp_list.append(second_part + "_" + pred_indices)
            else:
                tmp_list.append(second_part + "_0_" + str(len(source_utt.split()) - 1))
            new_data[int(conversation_id) - first_conv_id]["emotion-cause_pairs"].append(tmp_list)
            # print(str(len(source_utt.split()) - 1), " --> ", source_utt)
        i += 1

    # print(new_data)
    # Closing file
    f.close()

    # create output file
    out_filepath = "Subtask_1_pred.json"
    out_file = open(out_filepath, "w")
    # including video_name attribute which is not relevant for Subtask 1
    json.dump(new_data, out_file, indent=4)
    out_file.close()
    print(f"FINISHED: text span result saved to '{out_filepath}'")
