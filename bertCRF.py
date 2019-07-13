from os.path import join
from codecs import open
import torch
from torch.utils.data import Dataset
from pytorch_pretrained_bert import BertTokenizer,BertAdam,BertForTokenClassification
from vtb_bertcrf import vtb_pred
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.nn import CrossEntropyLoss
import pandas as pd
from sklearn.metrics import precision_score,recall_score,f1_score
from time import time

def build_corpus(split, make_vocab=True, data_dir=r"E:\ner\data"):
    """读取数据"""
    assert split in ['train', 'dev', 'test']
    word_lists = []
    tag_lists = []
    with open(join(data_dir, split+".char.bmes"), 'r', encoding='utf-8') as f:
        word_list = []
        tag_list = []
        for line in f:
            if line != '\n':
                word, tag = line.strip('\n').split()
                word_list.append(word.lower())
                tag_list.append(tag)
            else:
                word_lists.append(word_list)
                tag_lists.append(tag_list)
                word_list = []
                tag_list = []
    # 如果make_vocab为True，还需要返回word2id和tag2id
    if make_vocab:
#         word2id = build_map(word_lists)
        tag2id = build_map(tag_lists)
#         return word_lists, tag_lists, word2id, tag2id
        return word_lists, tag_lists,tag2id
    else:
        return word_lists, tag_lists
def build_map(lists):
    maps = {}
    for list_ in lists:
        for e in list_:
            if e not in maps:
                maps[e] = len(maps)
    return maps

word_lists, tag_lists, tag2id = build_corpus('train',make_vocab=True)
for i in range(len(word_lists)):
    # word_lists[i].append("<end>")
    tag_lists[i].append("<end>")
tag2id['<start>'] = len(tag2id)
tag2id['<end>'] = len(tag2id)

test_word_lists, test_tag_lists = build_corpus('test',make_vocab=False)



train = pd.DataFrame({'text':word_lists,'label':tag_lists})
test = pd.DataFrame({'text':test_word_lists,'label':test_tag_lists})

DEVICE = torch.device("cuda")
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

qlen = 185
class BulldozerDataset(Dataset):  # 集成Dataset，要重写3个函数 init，len，getitem
    def __init__(self, loadin_data, label_map):
        self.df = loadin_data
        self.label_map = label_map

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        '''根据IDX返回数据 '''
        tokens = self.df.iloc[idx, 0]  # list
        label = self.df.iloc[idx, 1]  # list

        if len(tokens) > qlen - 2:
            tokens = tokens[:qlen - 2]

        seq_word = ["[CLS]"] + tokens + ["[SEP]"]
        real_len = len(seq_word)

        ids = tokenizer.convert_tokens_to_ids(seq_word)
        ids_tensor = torch.tensor(ids)
        pad0 = torch.zeros(qlen - real_len).long()
        ids_tensor = torch.cat([ids_tensor, pad0])
        token_type_ids = torch.tensor([0] * qlen).long()
        attention_mask = torch.tensor([1] * real_len + [0] * (qlen - real_len)).long()

        # 将标签编码 tensor  未对标签pad填充
        label_indicate = [self.label_map.get(e) for e in label]
        if len(label_indicate) > qlen - 2:
            label_indicate = label_indicate[:qlen - 2]
        #
        label_indicate = torch.tensor(label_indicate + [-1] * (qlen - len(label_indicate)))

        return [ids_tensor, token_type_ids, attention_mask, label_indicate, real_len]

train_dataset = BulldozerDataset(train,tag2id)
test_dataset = BulldozerDataset(test,tag2id)
train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=0)
test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=True, num_workers=0)


class NER1(nn.Module):
    def __init__(self, num_labels):
        super(NER1, self).__init__()
        self.bert = BertForTokenClassification.from_pretrained('bert-base-chinese', num_labels=num_labels)
        for param in self.bert.parameters():
            param.requires_grad = True

        self.dp1 = nn.Dropout(0.1)

        self.transition = nn.Parameter(
            torch.ones(num_labels, num_labels) * 1 / num_labels)

    def forward(self, ids_tensor, token_type_ids, attention_mask):  # real_len是真实句长+2     b
        logits = self.bert(ids_tensor, token_type_ids, attention_mask)  # b*qlen*30

        batch_size, max_len, out_size = logits.size()
        crf_scores = logits.unsqueeze(
            2).expand(-1, -1, out_size, -1) + self.transition.unsqueeze(0)

        return crf_scores  # [B, L, out_size， out_size]


num_labels=len(tag2id)
torch.cuda.empty_cache()
ner1 = NER1(num_labels).to(DEVICE)
optimizer = BertAdam(filter(lambda p: p.requires_grad, ner1.parameters()),lr=0.00002)#(filter(lambda p: p.requires_grad, model4.parameters()),lr=0.001)
def cal_lstm_crf_loss(crf_scores, targets, tag2id): # crf_scores [B, L, out_size， out_size]     targets:带有end和pad的真实标签对应的id  tensor b*qlen
    """计算双向LSTM-CRF模型的损失
    该损失函数的计算可以参考:https://arxiv.org/pdf/1603.01360.pdf
    """
    pad_id = -1
    start_id = tag2id.get('<start>')
    end_id = tag2id.get('<end>')

    device = DEVICE

    # targets:[B, L] crf_scores:[B, L, T, T]
    batch_size, max_len = targets.size()  # 10*185
    target_size = len(tag2id)

    # mask = 1 - ((targets == pad_id) + (targets == end_id))  # [B, L]
    mask = (targets != pad_id)
    lengths = mask.sum(dim=1)  # batch里面每个句子长度

    # mask_bertout选出bertout中正确的部分，即去掉cls位置输出，除了句子和sep位置值为1，其他（包括cls位置）为0
    mask_bertout = torch.ones(mask.size()).float()
    for idx in range(len(mask)):
        mask_bertout[idx] = torch.cat([torch.tensor([0.]).to(DEVICE), mask[idx][:-1].float()])
    mask_bertout = (mask_bertout == 1).to(DEVICE)


    targets = indexed(targets, target_size, start_id)


    # # 计算Golden scores方法１
    # import pdb
    # pdb.set_trace()
    targets = targets.masked_select(mask)  # [real_L]

    flatten_scores = crf_scores.masked_select(
        mask_bertout.view(batch_size, max_len, 1, 1).expand_as(crf_scores)
    ).view(-1, target_size*target_size).contiguous()



    # maskbertout_crf_scores根据mask_bertout把除了文本和sep位置对应的矩阵之外的其他位置矩阵全变为0矩阵
    # new_score是将maskbertout_crf_scores进一步调整的结果，去掉maskbertout_crf_scores在cls位置0矩阵，再在最后补一个全0矩阵，用来计算all_path_scores
    maskbertout_crf_scores = crf_scores * mask_bertout.view(batch_size, max_len, 1, 1).expand_as(crf_scores).float()
    new_score = torch.ones(maskbertout_crf_scores.size()).float()
    for idx in range(len(maskbertout_crf_scores)):
        new_score[idx] = torch.cat([maskbertout_crf_scores[idx][1:], maskbertout_crf_scores[idx][0].unsqueeze(0)],
                                   dim=0)
        new_score = new_score.to(DEVICE)


    golden_scores = flatten_scores.gather(
        dim=1, index=targets.unsqueeze(1)).sum()

    scores_upto_t = torch.zeros(batch_size, target_size).to(DEVICE)
    # print('+++++++++++++')
    # print(lengths)
    for t in range(max_len):
        # 当前时刻 有效的batch_size（因为有些序列比较短)
        batch_size_t = (lengths > t).sum().item()
        if t == 0:
            scores_upto_t[:batch_size_t] = new_score[:batch_size_t,t, start_id, :]
        else:
            scores_upto_t[:batch_size_t] = torch.logsumexp(
                new_score[:batch_size_t, t, :, :] +
                scores_upto_t[:batch_size_t].unsqueeze(2),
                dim=1
            )
    all_path_scores = scores_upto_t[:, end_id].sum()

    # 训练大约两个epoch loss变成负数，从数学的角度上来说，loss = -logP
    loss = (all_path_scores - golden_scores) / batch_size
    # print(loss)
    return loss
def indexed(targets, tagset_size, start_id):
    """将targets中的数转化为在[T*T]大小序列中的索引,T是标注的种类"""
    batch_size, max_len = targets.size()
    for col in range(max_len-1, 0, -1):
        targets[:, col] += (targets[:, col-1] * tagset_size)
    targets[:, 0] += (start_id * tagset_size)
    return targets


torch.cuda.empty_cache()
F1_list = []
learn_str = time()
for epoch in range(5):
    epoch_str = time()
    ner1.train()
    for i, batchgroup in enumerate(train_iter):
        torch.cuda.empty_cache()  # 清除gpu缓存
        ids_tensor, token_type_ids, attention_mask = batchgroup[0].to(DEVICE), batchgroup[1].to(DEVICE), batchgroup[
            2].to(DEVICE)
        label_indicate, real_len = batchgroup[3].to(DEVICE), batchgroup[4].to(DEVICE)  # label_indicate   b*(qlen)

        real_len, sort_idx = torch.sort(real_len, descending=True)
        ids_tensor, token_type_ids, attention_mask,label_indicate = ids_tensor[sort_idx], token_type_ids[sort_idx], attention_mask[sort_idx],label_indicate[sort_idx]

        scores = ner1(ids_tensor, token_type_ids, attention_mask)
        optimizer.zero_grad()
        loss = cal_lstm_crf_loss(scores, label_indicate, tag2id).to(DEVICE)
        # print(loss)
        loss.backward()
        optimizer.step()
    print('epoch:',epoch+1, 'loss:',loss)
    epoch_time = time()-epoch_str




    ner1.eval()
    eval_str = time()
    with torch.no_grad():
        predtag_list = []
        truetag_list = []
        for i, batchgroup in enumerate(test_iter):
            torch.cuda.empty_cache()  # 清除gpu缓存
            ids_tensor, token_type_ids, attention_mask = batchgroup[0].to(DEVICE), batchgroup[1].to(DEVICE), batchgroup[
                2].to(DEVICE)
            label_indicate, real_len = batchgroup[3].to(DEVICE), batchgroup[4].to(DEVICE)  # label_indicate   b*(qlen)

            real_len, sort_idx = torch.sort(real_len, descending=True)
            ids_tensor, token_type_ids, attention_mask, label_indicate = ids_tensor[sort_idx], token_type_ids[sort_idx], \
                                                                         attention_mask[sort_idx], label_indicate[
                                                                             sort_idx]
            words_withEND_len = real_len-1  # 减去cls位置才是加上sep的正真长度
            scores = ner1(ids_tensor, token_type_ids, attention_mask)
            # 去掉score的cls位置，在每个batch的末尾位置补0矩阵
            mask_bertout = torch.zeros(scores.size()).to(DEVICE)
            for idx in range(len(words_withEND_len)):
                mask_bertout[idx][1:1+words_withEND_len[idx]] += 1
            mask_bertout = (mask_bertout == 1)
            maskbertout_crf_scores = scores * mask_bertout.float()
            new_score = torch.ones(maskbertout_crf_scores.size()).float()
            for idx in range(len(maskbertout_crf_scores)):
                new_score[idx] = torch.cat(
                    [maskbertout_crf_scores[idx][1:], maskbertout_crf_scores[idx][0].unsqueeze(0)],
                    dim=0)
                new_score = new_score.to(DEVICE)

            tagids =vtb_pred(words_withEND_len, tag2id,new_score,DEVICE)

            mask_tagids = (tagids!=-1)
            pretag_for_batch_1D = tagids.masked_select(mask_tagids)
            pretag_for_batch_list = list(pretag_for_batch_1D.cpu().numpy())
            predtag_list+=pretag_for_batch_list

            mask_label = (label_indicate != -1)
            label_for_batch_1D = label_indicate.masked_select(mask_label)
            label_for_batch_list = list(label_for_batch_1D.cpu().numpy())
            truetag_list += label_for_batch_list

        print('avg准确率：', precision_score(truetag_list, predtag_list, average='macro'))
        print('avg召回率：', recall_score(truetag_list, predtag_list, average='macro'))
        f1 = f1_score(truetag_list, predtag_list, average='macro')
        F1_list.append((f1))
        print('avgF1：', f1)
        eval_time = time() - eval_str
        print('eval耗时（min）：', eval_time / 60)
        print('epoch训练耗时（min）：', epoch_time / 60)
        print('\n')


print(F1_list)
learn_time = time()-learn_str
print('train耗时（min）：',learn_time/60)





            # print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
            # print(epoch,i,tagids.size())
            # print(tagids)
            # print(label_indicate)
            # print('sun+++')
            # print(torch.sum((tagids!=-1),dim=1))
            # print(torch.sum((label_indicate != -1), dim=1))
            # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')






#     eval_str = time()
#     with torch.no_grad():
#         pre_label = []
#         tar_label = []
#         for i, batchgroup in enumerate(test_iter):
#             torch.cuda.empty_cache()  # 清除gpu缓存
#             ids_tensor, token_type_ids, attention_mask = batchgroup[0].to(DEVICE), batchgroup[1].to(DEVICE), batchgroup[
#                 2].to(DEVICE)
#             label_indicate, real_len = batchgroup[3].to(DEVICE), batchgroup[4].to(DEVICE)
#             logits, new_attention_mask = ner1(ids_tensor, token_type_ids, attention_mask, real_len)
#             pred = logits.max(2, keepdim=True)[1]  # pred  b*(qlen-2)*1
#
#             active_loss = new_attention_mask.view(-1) == 1
#             pre_rensor = pred.view(-1)[active_loss]  # .to(DEVICE)
#             tar_tensor = label_indicate.view(-1)[active_loss]
#
#             pre_list = list(pre_rensor.cpu().numpy())
#             tar_list = list(tar_tensor.cpu().numpy())
#             pre_label += pre_list
#             tar_label += tar_list
#
#     print('avg准确率：', precision_score(tar_label, pre_label, average='macro'))
#     print('avg召回率：', recall_score(tar_label, pre_label, average='macro'))
#     f1 = f1_score(tar_label, pre_label, average='macro')
#     F1_list.append((f1))
#     print('avgF1：', f1)
#     eval_time = time() - eval_str
#     print('eval耗时（min）：', eval_time / 60)
#     print('epoch耗时（min）：', epoch_time / 60)
#     print('\n')
#
#
# print(F1_list)
#
#
#
#
#
#
#
#
#
#
#
#
# learn_time = time()-learn_str
# print('train耗时（min）：',learn_time/60)



