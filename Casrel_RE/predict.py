# coding:utf-8
import json

import torch
from Casrel_RE.config import *
from model.CasrelModel import *
from utils.process import *
conf = Config()

def load_model(model_path):
    model = Casrel(conf)
    model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
    return model

def get_inputs(sample, model):
    '''
    获取样本的输入
    '''
    print(f'sample--》{sample}')
    text = conf.tokenizer(sample)
    print(f'text--》{text}')
    input_ids = torch.tensor([text['input_ids']]) # [1, 13]
    mask = torch.tensor([text['attention_mask']]) # [1, 13]
    seq_len = len(text["input_ids"])
    inner_sub_head2tail = torch.zeros(seq_len)
    print(f'inner_sub_head2tail--》{inner_sub_head2tail}')
    inner_sub_len = torch.tensor([1], dtype=torch.float)
    # 预测主实体
    model.eval()
    with torch.no_grad():
        # 得到bert模型的编码结果[1, 13, 768]
        encode_text = model.get_encoded_text(input_ids, mask)
        # 预测主实体的开始和结束位置pred_sub_heads, pred_sub_tails--<>shape-->[1, 13, 1]
        pred_sub_heads, pred_sub_tails = model.get_subs(encode_text)
    # 将预测 结果进行01转换
    pred_sub_heads = convert_score_to_zero_one(pred_sub_heads)
    pred_sub_tails = convert_score_to_zero_one(pred_sub_tails)
    # 抽取出主实体的开始和结束位置
    pred_subs = extract_sub(pred_sub_heads.squeeze(), pred_sub_tails.squeeze())
    # print(f'pred_subs--》{pred_subs}')
    if pred_subs:
        sub_head_idx = pred_subs[0][0]
        sub_tail_idx = pred_subs[0][1]
        inner_sub_head2tail[sub_head_idx: sub_tail_idx+1] = 1
        inner_sub_len = torch.tensor([sub_tail_idx+1-sub_head_idx], dtype=torch.float)
    sub_len = inner_sub_len.unsqueeze(0) # [1, 1]
    sub_head2tail = inner_sub_head2tail.unsqueeze(0) # [1, 13]
    inputs = {'input_ids': input_ids,
              'mask': mask,
              'sub_head2tail': sub_head2tail,
              'sub_len': sub_len}
    return inputs, model

def model2predict(sample, model):
    # 读取关系字典
    with open(conf.rel_dict_path, 'r', encoding='utf-8') as fr:
        id2rel = json.load(fr)
    # print(f'id2rel-=》{id2rel}')
    inputs, model = get_inputs(sample, model)
    # print(f'inputs--》{inputs}')
    logist = model(**inputs)
    # print(f"logist['pred_sub_heads']-->{logist['pred_sub_heads'].shape}")
    # print(f"logist['pred_obj_heads']-->{logist['pred_obj_heads'].shape}")
    pred_sub_heads = convert_score_to_zero_one(logist['pred_sub_heads'])
    pred_sub_tails = convert_score_to_zero_one(logist['pred_sub_tails'])
    pred_obj_heads = convert_score_to_zero_one(logist['pred_obj_heads'])
    pred_obj_tails = convert_score_to_zero_one(logist['pred_obj_tails'])
    new_dict = {}
    spo_list = []
    # 获取输入的ids
    ids = inputs["input_ids"][0]
    # print(f'ids--》{ids}')
    token_list = conf.tokenizer.convert_ids_to_tokens(ids)
    # print(f'token_list==>{token_list}')
    sentence = ''.join(token_list[1:-1])
    # print(f'sentence--=》{sentence}')
    pred_subs = extract_sub(pred_sub_heads[0].squeeze(), pred_sub_tails[0].squeeze())
    # print('*'*80)
    # print(f'pred_subs--》{pred_subs}')
    # pred_obj_heads-->[1, 13, 18]; pred_obj_tails-->[1, 13, 18]
    pred_objs = extract_obj_and_rel(pred_obj_heads[0], pred_obj_tails[0])
    # print(f'pred_objs--》{pred_objs}')
    if len(pred_subs) == 0 or len(pred_objs) == 0:
        print('没有识别出结果')
        return {}
    if len(pred_objs) > len(pred_subs):
        pred_subs = pred_subs*len(pred_objs)
    for sub, rel_obj in zip(pred_subs, pred_objs):
        # print(f'sub--》{sub}')
        # print(f'rel_obj--》{rel_obj}')
        sub_spo = {}
        sub_head_idx, sub_tail_idx = sub
        sub_str = ''.join(token_list[sub_head_idx: sub_tail_idx+1])
        # print(f'sub_str--》{sub_str}')
        if '[PAD]' in sub_str:
            continue
        sub_spo["subject"] = sub_str
        # print(f'sub_spo--》{sub_spo}')
        relation = id2rel[str(rel_obj[0])]
        # print(f'relation--》{relation}')
        obj_head, obj_tail = rel_obj[1], rel_obj[2]
        obj_str = ''.join(token_list[obj_head: obj_tail + 1])
        if '[PAD]' in obj_str:
            continue
        # print(f'obj_str--》{obj_str}')
        sub_spo['predicate'] = relation
        sub_spo['object'] = obj_str
        spo_list.append(sub_spo)

        # break

    new_dict['text'] = sentence
    new_dict['spo_list'] = spo_list
    return new_dict

if __name__ == '__main__':
    # sample = "《人间》是王菲演唱歌曲"
    # sample = "刘冬元，(1953－1992)中共党员，祁阳县凤凰乡凤凰村人，1953年11月出生，1969年参加工作，先后任凤凰公社话务员、广播员，上司源乡中学副校长，白果市乡中学校长、辅导区主任、金洞学区业务专干、百里乡人民政府纪检员"
    sample = '《红海大漠》是中国青年出版社出版的图书，作者是梁子'
    model_path = r'D:\AI_agent\Medical_Graph\Casrel_RE\save_model\last_model.pth'
    model = load_model(model_path)
    print(f'model--》{model}')
    result = model2predict(sample, model)
    print(result)