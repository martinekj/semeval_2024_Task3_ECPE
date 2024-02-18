import config
import os
import json
import jsonpickle
from EmotionCauseLink import EmotionCauseLink

from UtteranceItem import UtteranceItem

import utils


def load_dialogues_json_data(dataset_type="train"):
    nodes = {}
    filepath = config.train_test_dev_splits_path+dataset_type+".json"
    with open(filepath, mode="r", encoding="utf8") as file:
        data = jsonpickle.decode(file.read())

    for key, item in data.items():
        # node_id format = <CONVERSATION_ID>_<UTTERANCE_ID>
        node_id = key
        utt_node = UtteranceItem(node_id, item["utterance_text"], item["utterance_emotion"], item["utterance_speaker"], item["video_name"])
        utt_node.emotion_cause_links = item["emotion_cause_links"]
        if node_id not in nodes:
            nodes[node_id] = utt_node
        else:
            print(f"Error: duplicite utterance id {node_id}")

    return nodes


def prepare_eval_data(output_filename):
    # load all data
    # Opening JSON file
    f = open(config.eval_data_path)

    # returns JSON object as
    # a dictionary
    data = json.load(f)

    # Closing file
    f.close()

    eval_nodes = {}
    # create a graph from data --> utterances are nodes, emotion-cause pairs = an edge between two nodes
    for item in data:
        conversation = item["conversation"]
        for utt in conversation:
            # node_id format = <CONVERSATION_ID>_<UTTERANCE_ID>
            node_id = str(item['conversation_ID'])+"_"+str(utt['utterance_ID'])
            utt_node = UtteranceItem(node_id, utt['text'], "", utt['speaker'], utt["video_name"])
            if node_id not in eval_nodes:
                eval_nodes[node_id] = utt_node
            else:
                print(f"Error: duplicite utterance id {node_id}")

    # output_filename = os.path.join(config.train_test_dev_splits_path, "eval.json")
    utils.create_json_representation_file(eval_nodes, output_filename=output_filename)


