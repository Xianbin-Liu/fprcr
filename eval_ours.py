import numpy as np
import os
import pandas as pd
import sys
import argparse
import subprocess

def recommend_eval(users, items, topk_items, top_k):
    hits = []
    recalls = []
    ndcgs = []
    accs = []

    for i, user in enumerate(users):
        label = items[user.item()]
        pred = topk_items[i].tolist()[:top_k]
        hits.append(len(set(label).intersection(set(pred))))
        recalls.append(hits[i] / len(label))
        accs.append(1 if hits[-1] > 0 else 0)

        dcg = 0.0
        idcg = 0.0
        for rank, item in enumerate(pred):
            if item in label:
                dcg += 1.0 / np.log2(rank + 2)
        for rank in range(min(len(label), top_k)):
            idcg += 1.0 / np.log2(rank + 2)
        ndcgs.append(dcg / idcg)

    precision = np.mean(hits) / top_k
    recall = np.mean(recalls)
    f1 = 2 * precision * recall / (precision + recall)
    ndcg = np.mean(ndcgs)
    acc = np.mean(accs)

    return precision, recall, f1, ndcg, acc
def recommend_eval_gcnn(data, top_k, correct=False):
    hits = []
    recalls = []
    ndcgs = []
    accs = []

    total_test_item = 14239 - 1
    filter_test_item = len(data)

    for i in range(len(data)):
        label = data.iloc[i]['gt']
        pred = data.iloc[i]['pred'][:top_k]

        hits.append(len(set(label).intersection(set(pred))))
        recalls.append(hits[i] / len(label))
        accs.append(1 if hits[-1] > 0 else 0)

        dcg = 0.0
        idcg = 0.0
        for rank, item in enumerate(pred):
            if item in label:
                dcg += 1.0 / np.log2(rank + 2)
        for rank in range(min(len(label), top_k)):
            idcg += 1.0 / np.log2(rank + 2)
        ndcgs.append(dcg / idcg)

    if correct:
        acc = np.sum(accs) / total_test_item
        precision = np.mean(hits) / top_k
        recall = np.mean(recalls)
        f1 = 2 * precision * recall / (precision + recall)
        ndcg = np.mean(ndcgs)
    else:
        precision = np.mean(hits) / top_k
        recall = np.mean(recalls)
        f1 = 2 * precision * recall / (precision + recall)
        ndcg = np.mean(ndcgs)
        acc = np.mean(accs)

    return precision, recall, f1, ndcg, acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='log/20240617100047')
    args = parser.parse_args()

    log_dir = args.log_dir
    pred_path = os.path.join(log_dir, 'top_10_predictions_smi.csv')
    result_path = os.path.join(log_dir, 'top_10_predictions_result.txt')

    correct_with_total_item = False
    # pred_path = './uspto_results/uspto_final_predictions_top20.csv'
    data = pd.read_csv(pred_path).reset_index(drop=True)
    data['pred'] = data['pred_label'].apply(lambda x: eval(x))
    data['gt'] = data['gt_label'].apply(lambda x: eval(x))

    # print(len(data['gt'].unique()))

    with open(result_path, 'w') as f:
        top_ks = [1, 3, 5, 10, 20]
        for top_k in top_ks:
            
            precision, recall, f1, ndcg, acc = recommend_eval_gcnn(data, top_k, correct_with_total_item)
            print('top k: ', top_k)
            print('precision: ', "{:.4f}".format(precision))
            print('recall: ', "{:.4f}".format(recall))
            print('f1: ', "{:.4f}".format(precision))
            print('ndcg: ', "{:.4f}".format(ndcg))
            if correct_with_total_item:
                print('corrected acc: ', "{:.4f}".format(acc))
                f.write(f'top k: {top_k}\nprecision: {precision}\nrecall: {recall}\nf1: {f1}\nndcg: {ndcg}\ncorrected acc: {acc}\n\n')
            else:
                print('acc: ', "{:.4f}".format(acc))
                f.write(f'top k: {top_k}\nprecision: {precision}\nrecall: {recall}\nf1: {f1}\nndcg: {ndcg}\nacc: {acc}\n\n')



## result
# top k:  1
# precision:  0.0854
# recall:  0.0143
# f1:  0.0854
# ndcg:  0.2495
# acc:  0.0854
# top k:  3
# precision:  0.1188
# recall:  0.0571
# f1:  0.1188
# ndcg:  0.2685
# acc:  0.3563
# top k:  5
# precision:  0.0981
# recall:  0.0783
# f1:  0.0981
# ndcg:  0.2463
# acc:  0.4904
# top k:  10
# precision:  0.0575
# recall:  0.0909
# f1:  0.0575
# ndcg:  0.2275
# acc:  0.5749
# top k:  20
# precision:  0.0290
# recall:  0.0918
# f1:  0.0290
# ndcg:  0.2429
# acc:  0.5803