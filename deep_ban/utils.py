import dill
import torch
from config import config_1, config_2, config_3

from model import BilstmCrf, create_model
import jieba
from operator import itemgetter

with open('data/model_1/src.pkl', 'rb') as F:
    SRC_1 = dill.load(F)

with open('data/model_2/src.pkl', 'rb') as F:
    SRC_2 = dill.load(F)

with open('data/model_3/src_label2.pkl', 'rb') as F:
    src_label = dill.load(F)

config1 = config_1.Config()
config2 = config_2.Config()
config3 = config_3.Config()
config3.LABEL = src_label['label']
config3.SRC = src_label['src']

model_1 = create_model(config1, SRC_1)
model_2 = create_model(config2, SRC_2)
model_3 = BilstmCrf(config3)

model_1.load_state_dict(torch.load('data/model_1/rnn.h5', map_location=lambda storage, loc: storage))
model_2.load_state_dict(torch.load('data/model_2/rnn.h5', map_location=lambda storage, loc: storage))
model_3.load_state_dict(torch.load('data/model_3/bilstm_crf2.h5', map_location=lambda storage, loc: storage))


def intention_recognition(model, SRC, inputs, seq_len):
    model.eval()
    inputs = jieba.lcut(inputs)
    inputs += ['<pad>'] * (seq_len -len(inputs))
    inputs = itemgetter(*inputs)(SRC.vocab.stoi)
    inputs = torch.tensor([inputs])
    if len(inputs.shape) == 1:
        inputs = inputs.unsqueeze(0)
    _, index = torch.max(model.forward(inputs), dim=-1)
    return index.item()


def predict_ner(model, SRC, LABEL, inputs):
    """
    """
    model.eval()
    res = itemgetter(*inputs)(SRC.vocab.stoi)
    res = torch.tensor(res).unsqueeze(0)
    answers = model.decode(res)

    extracted_entities = extract(answers[0], LABEL.vocab.itos)
    L = []
    for extracted_entity in extracted_entities:
        start_index = int(extracted_entity['start_index'])
        end_index = int(extracted_entity['end_index']) + 1
        entity = {'content': inputs[start_index: end_index], 'label': extracted_entity['name'],
                  'index': [start_index, end_index]}
        L.append(entity)

    return L, inputs


def extract(answer, idx_to_label):
    # idx_to_label = {k: v for k, v in enumerate(idx_to_label)}
    answer = itemgetter(*answer)(idx_to_label)
    extracted_entities = []
    current_entity = None
    for index, label in enumerate(answer):
        if label in ['O', '<pad>']:
            if current_entity:
                current_entity = None
                continue
            else:
                continue
        else:
            # position  B I E S
            position, entity_type = label.split('-')
            if current_entity:
                if entity_type == current_entity['name']:
                    if position == 'S':
                        extracted_entities.append({
                            'name': entity_type, 'start_index': index, 'end_index': index
                        })
                        current_entity = None
                    elif position == 'I':
                        continue
                    elif position == 'B':
                        current_entity = {
                            'name': entity_type, 'start_index': index, 'end_index': None
                        }
                        continue
                    else:
                        current_entity['end_index'] = index
                        extracted_entities.append(current_entity)
                        current_entity = None


                else:
                    if position == 'S':
                        extracted_entities.append({
                            'name': entity_type, 'start_index': index, 'end_index': index
                        })
                        current_entity = None
                    if position == 'B':
                        current_entity = {
                            'name': entity_type, 'start_index': index, 'end_index': None
                        }

            else:
                if position == 'S':
                    extracted_entities.append({
                        'name': entity_type, 'start_index': index, 'end_index': index
                    })
                    current_entity = None
                if position == 'B':
                    current_entity = {
                        'name': entity_type, 'start_index': index, 'end_index': None
                    }

    return extracted_entities


def rebuildiins(ins, entity_list):
    ins = list(ins)
    for entity in entity_list:
        start_index, end_index = entity['index']
        ins[start_index:end_index] = entity['label']
    return ''.join(ins)


if __name__ == '__main__':
    # inputs = ''.join(["???", "???", "???", "???", "???", "???", "???", "???", "???", "???", "???", "???"])
    inputs = ''.join(["???", "???", "???", "???", "???", '?????????'])
    inputs = '?????????????????????????????????'
    # inputs = ''.join(["???", "???", "???", "???", "???", "???", "???", "???", "???", "???", "???", "???", "???"])
    inputs = '?????????????????????'
    inputs = '????????????'
    inputs = '??????????????????????????????????????????'
    inputs = '?????????????????????????????????'
    entitys = predict_ner(model_3, src_label['src'], src_label['label'], inputs)
    print(entitys)
    entity_list, ins = entitys

    print(rebuildiins(ins, entity_list))
