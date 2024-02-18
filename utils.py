import jsonpickle
from UtteranceItem import UtteranceItem
import os


def create_json_representation_file(nodes, output_filename):
    # Printing and saving the dictionary data
    jsonpickle.set_encoder_options('json', sort_keys=False, indent=4)
    json_string = jsonpickle.encode(nodes, unpicklable=False)
    # if os.path.exists(output_filename):
    #     raise FileExistsError("File already exists!")
    with open(output_filename, encoding="utf8", mode="w") as fw:
        fw.write(json_string)
    print(f"Filename {output_filename} created")


def create_semeval_format(source_json_file, target_json_filename, trial=False):
    #print(all_train_data)
    with open(source_json_file, mode="r", encoding="utf8") as file:
        source_data = jsonpickle.decode(file.read())

    #print(gt_data)
    all_data = []
    dialogue_ids_processed = []
    #print(dialogues_ids)
    for key in source_data.keys():
        if trial == True:
            key_dialogue_part = int(key.split("_")[1])
            key_utterance_part = int(key.split("_")[2])
        else:
            key_dialogue_part = int(key.split("_")[0])
            key_utterance_part = int(key.split("_")[1])
        if key_dialogue_part not in dialogue_ids_processed:
            dict = {}
            dict["conversation_ID"] = key_dialogue_part
            dict["conversation"] = []
            dict["emotion-cause_pairs"] = []
            all_data.append(dict)
            dialogue_ids_processed.append(key_dialogue_part)

        dict = all_data[-1]

        dict["conversation"].append({
            'utterance_ID': key_utterance_part,
            'text': source_data[key]["utterance_text"],
            'speaker': source_data[key]["utterance_speaker"],
            'emotion': source_data[key]["utterance_emotion"],
            'video_name': source_data[key]["video_name"]
        })
        for link in source_data[key]["emotion_cause_links"]:
            lst = []
            if trial == True:
                part1 = str(link["target_id"].split("_")[2]) + "_" + link["emotion"]
                part2 = str(link["source_id"].split("_")[2])
            else:
                part1 = str(link["target_id"].split("_")[1]) + "_" + link["emotion"]
                part2 = str(link["source_id"].split("_")[1])
            lst.append(part1)
            lst.append(part2)
            dict["emotion-cause_pairs"].append(lst)


    # Printing and saving the dictionary data
    jsonpickle.set_encoder_options('json', sort_keys=False, indent=4)
    json_string = jsonpickle.encode(all_data, unpicklable=False)
    with open(target_json_filename, encoding="utf8", mode="w") as fw:
        fw.write(json_string)
    print(f"Filename {target_json_filename} created")


def load_dialogues_json_data(filename):
    data = {}
    with open(filename, mode="r", encoding="utf8") as file:
        json_data = jsonpickle.decode(file.read())

    for key, item in json_data.items():
        # node_id format = <CONVERSATION_ID>_<UTTERANCE_ID>
        node_id = key
        utt_node = UtteranceItem(node_id, item["utterance_text"], item["utterance_emotion"], item["utterance_speaker"], item["video_name"])
        utt_node.emotion_cause_links = item["emotion_cause_links"]
        if node_id not in data:
            data[node_id] = utt_node
        else:
            print(f"Error: duplicite utterance id {node_id}")

    return data


def utterances_to_dialogues(utts):
    res = {}
    for k, v in utts.items():
        uttid = k.split("_")[0]
        if uttid not in res:
            res[uttid] = []

        res[uttid].append(v)
    return res


def dialogues_to_utterances(dials):
    res = {}
    for k, v in dials.items():
        for utt in v:
            res[utt.utterance_id] = utt
    return res
