import torch
from itertools import zip_longest

def vtb_pred(lengths, tag2id,crf_scores,DEVICE):
    """使用维特比算法进行解码"""
    start_id = tag2id['<start>']
    end_id = tag2id['<end>']
    pad = -1
    tagset_size = len(tag2id)

    # crf_scores = self.forward(test_sents_tensor, lengths)
    device = DEVICE
    # B:batch_size, L:max_len, T:target set size
    B, L, T, _ = crf_scores.size()
    # viterbi[i, j, k]表示第i个句子，第j个字对应第k个标记的最大分数
    viterbi = torch.zeros(B, L, T).to(device)
    # backpointer[i, j, k]表示第i个句子，第j个字对应第k个标记时前一个标记的id，用于回溯
    backpointer = (torch.zeros(B, L, T).long() * end_id).to(device)
    # lengths = torch.LongTensor(lengths).to(device)
    # 向前递推
    # print('lengths:',lengths)
    L = lengths[0]
    for step in range(L):
        batch_size_t = (lengths > step).sum().item()
        if step == 0:
            # 第一个字它的前一个标记只能是start_id
            viterbi[:batch_size_t, step,
            :] = crf_scores[: batch_size_t, step, start_id, :]
            backpointer[: batch_size_t, step, :] = start_id
        else:
            # x = viterbi[:batch_size_t, step -1, :].unsqueeze(2)+crf_scores[:batch_size_t, step, :, :]
            # # print(x)
            # print('<><><><>><><><><><><><><><><><><><><><><><><><>><><><><')
            # print(step,x.size(),batch_size_t)
            max_scores, prev_tags = torch.max(
                viterbi[:batch_size_t, step -1, :].unsqueeze(2) +
                crf_scores[:batch_size_t, step, :, :],     # [B, T, T]
                dim=1
            )
            viterbi[:batch_size_t, step, :] = max_scores
            backpointer[:batch_size_t, step, :] = prev_tags

    # 在回溯的时候我们只需要用到backpointer矩阵
    backpointer = backpointer.view(B, -1)  # [B, L * T]
    tagids = []  # 存放结果
    tags_t = None
    for step in range( L -1, 0, -1):
        batch_size_t = (lengths > step).sum().item()
        if step == L- 1:
            index = torch.ones(batch_size_t).long() * (step * tagset_size)
            index = index.to(device)
            index += end_id
        else:
            prev_batch_size_t = len(tags_t)

            new_in_batch = torch.LongTensor([end_id] * (batch_size_t - prev_batch_size_t)).to(device)
            offset = torch.cat(
                [tags_t, new_in_batch],
                dim=0
            )  # 这个offset实际上就是前一时刻的
            index = torch.ones(batch_size_t).long() * (step * tagset_size)
            index = index.to(device)
            index += offset.long()

        try:
            tags_t = backpointer[:batch_size_t].gather(
                dim=1, index=index.unsqueeze(1).long())
        except RuntimeError:
            import pdb
            pdb.set_trace()
        tags_t = tags_t.squeeze(1)
        tagids.append(tags_t.tolist())

    # tagids:[L-1]（L-1是因为扣去了end_token),大小的liebiao
    # 其中列表内的元素是该batch在该时刻的标记
    # 下面修正其顺序，并将维度转换为 [B, L]
    tagids = list(zip_longest(*reversed(tagids), fillvalue=pad))
    tagids = torch.Tensor(tagids).long()

    # 返回解码的结果
    return tagids