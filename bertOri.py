from os.path import join
from codecs import open
import torch
from torch.utils.data import Dataset
from pytorch_pretrained_bert import BertTokenizer,BertAdam,BertForTokenClassification
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
test_word_lists, test_tag_lists = build_corpus('test',make_vocab=False)

train = pd.DataFrame({'text':word_lists,'label':tag_lists})
test = pd.DataFrame({'text':test_word_lists,'label':test_tag_lists})

DEVICE = torch.device("cuda")
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

qlen = 180
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
        # 注意  label_indicate只需补偿到qlen-2长度 因为bert后输出结果是qlen长度，但要减掉cls和sep两个长度 so正真输出长度是qlen-2
        label_indicate = torch.tensor(label_indicate + [-1] * (qlen - 2 - len(label_indicate)))

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

    def forward(self, ids_tensor, token_type_ids, attention_mask, real_len):  # real_len是真实句长+2     b
        logits = self.bert(ids_tensor, token_type_ids, attention_mask)  # b*qlen*28

        # 去掉cls和sep位置的输出
        new_logits = torch.ones(logits.size(0), logits.size(1) - 2, logits.size(2))
        new_attention_mask = torch.ones(attention_mask.size(0), attention_mask.size(1) - 2)
        for idx in range(len(logits)):
            sentlen = real_len[idx] - 2
            new_logits[idx] = torch.cat([logits[idx][1:(sentlen + 1), :], logits[idx][(sentlen + 2):, :]], dim=0)
            new_attention_mask[idx] = torch.cat(
                [attention_mask[idx][1:(sentlen + 1)], attention_mask[idx][(sentlen + 2):]], dim=0)

        return new_logits, new_attention_mask  # logits b*(qlen-2)*28   attention_mask  b*(qlen-2)


num_labels=28
torch.cuda.empty_cache()
ner1 = NER1(num_labels).to(DEVICE)
optimizer = BertAdam(filter(lambda p: p.requires_grad, ner1.parameters()),lr=0.00002)#(filter(lambda p: p.requires_grad, model4.parameters()),lr=0.001)
loss_funtion = CrossEntropyLoss()


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
        label_indicate, real_len = batchgroup[3].to(DEVICE), batchgroup[4].to(DEVICE)  # label_indicate   b*(qlen-2)

        logits, new_attention_mask = ner1(ids_tensor, token_type_ids, attention_mask, real_len)

        active_loss = new_attention_mask.view(-1) == 1
        active_logits = logits.view(-1, num_labels)[active_loss].to(DEVICE)
        active_labels = label_indicate.view(-1)[active_loss]

        optimizer.zero_grad()
        loss = loss_funtion(active_logits, active_labels)
        loss.backward()
        optimizer.step()
    print('epoch:',epoch+1, 'loss:',loss)
    epoch_time = time()-epoch_str


    ner1.eval()
    eval_str = time()
    with torch.no_grad():
        pre_label = []
        tar_label = []
        for i, batchgroup in enumerate(test_iter):
            torch.cuda.empty_cache()  # 清除gpu缓存
            ids_tensor, token_type_ids, attention_mask = batchgroup[0].to(DEVICE), batchgroup[1].to(DEVICE), batchgroup[
                2].to(DEVICE)
            label_indicate, real_len = batchgroup[3].to(DEVICE), batchgroup[4].to(DEVICE)
            logits, new_attention_mask = ner1(ids_tensor, token_type_ids, attention_mask, real_len)
            pred = logits.max(2, keepdim=True)[1]  # pred  b*(qlen-2)*1

            active_loss = new_attention_mask.view(-1) == 1
            pre_rensor = pred.view(-1)[active_loss]  # .to(DEVICE)
            tar_tensor = label_indicate.view(-1)[active_loss]

            pre_list = list(pre_rensor.cpu().numpy())
            tar_list = list(tar_tensor.cpu().numpy())
            pre_label += pre_list
            tar_label += tar_list

    print('avg准确率：', precision_score(tar_label, pre_label, average='macro'))
    print('avg召回率：', recall_score(tar_label, pre_label, average='macro'))
    f1 = f1_score(tar_label, pre_label, average='macro')
    F1_list.append((f1))
    print('avgF1：', f1)
    eval_time = time() - eval_str
    print('eval耗时（min）：', eval_time / 60)
    print('epoch耗时（min）：', epoch_time / 60)
    print('\n')


print(F1_list)









learn_time = time()-learn_str
print('train耗时（min）：',learn_time/60)



# ner1.eval()
# eval_str = time()
# with torch.no_grad():
#     pre_label = []
#     tar_label = []
#     for i, batchgroup in enumerate(train_iter):
#         torch.cuda.empty_cache()  # 清除gpu缓存
#         ids_tensor, token_type_ids, attention_mask = batchgroup[0].to(DEVICE), batchgroup[1].to(DEVICE), batchgroup[
#             2].to(DEVICE)
#         label_indicate, real_len = batchgroup[3].to(DEVICE), batchgroup[4].to(DEVICE)
#         logits, new_attention_mask = ner1(ids_tensor, token_type_ids, attention_mask, real_len)
#         pred = logits.max(2, keepdim=True)[1]  #pred  b*(qlen-2)*1
#
#
#         active_loss = new_attention_mask.view(-1) == 1
#         pre_rensor = pred.view(-1)[active_loss]#.to(DEVICE)
#         tar_tensor = label_indicate.view(-1)[active_loss]
#
#         pre_list = list(pre_rensor.cpu().numpy())
#         tar_list = list(tar_tensor.cpu().numpy())
#         pre_label+=pre_list
#         tar_label+=tar_list
#
# print('avg准确率：',precision_score(tar_label, pre_label, average='macro'))
# print('avg召回率：',recall_score(tar_label, pre_label, average='macro'))
# print('avgF1：',f1_score(tar_label, pre_label, average='macro'))
# eval_time = time()-eval_str
# print('eval耗时（min）：',eval_time/60)


