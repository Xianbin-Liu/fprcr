import heapq
from collections import OrderedDict
import pandas as pd
import json
import torch
import argparse

with open('./data/uspto_dict/label2bin_uni.json', 'r') as fp:
    id2uni = json.load(fp)

with open('./data/uspto_dict/bins2label.json', 'r') as fp:
    id2name = json.load(fp)


parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', type=str, default='log/20240617100047')
args = parser.parse_args()

log_dir = args.log_dir
data_dir = 'data/uspto_condition_split'
# torch.load 
pred = torch.load(log_dir + '/raw_output.pth')

pred = {
    'A': pred['c'].numpy(),
    'B': pred['s1'].numpy(),
    'C': pred['s2'].numpy(),
    'D': pred['r1'].numpy(),
    'E': pred['r2'].numpy()
}

data_points = pred['A'].shape[0]


truth_file = data_dir + '/test.csv'
truth = pd.read_csv(truth_file, header=0)
columns = ['catalyst1_bin', 'solvent1_bin', 'solvent2_bin', 'reagent1_bin', 'reagent2_bin']

predictions = []

assert len(truth) == data_points

num_ranks = 10
class_dict = OrderedDict({'A': 54, 'B': 85, 'C':41, 'D': 223, 'E': 95})
keys = ['A', 'B', 'C', 'D', 'E']

# read the prediction results, shape: (nums_item, nums_class)
# each item: 0~52: class: A; 54~138: class: B; 140~180: class: C; 182~404: class: D; 406~500: class: E

# find the top-20 predictions
# first we split the results
class max_heap:
    def __init__(self):
        self.data = []
    
    def __repr__(self) -> str:
        return self.data
    
    def push(self, item):
        heapq.heappush(self.data, (-item[0], item[1]))

    def pop(self):
        item = heapq.heappop(self.data)
        return (-item[0], item[1])

    def top(self):
        return (-self.data[0][0], self.data[0][1])

def find_top_k(pred, k=10):
    # @praram pred: {A:[prob., size of 54], B:[prob., size of 85], C:[prob., size of 41], D:[prob., size of 223], E:[prob., size of 95]}
    # @return: top-k predictions
    def get_indices(pred_sort, idx):
        indices = []
        for i in range(len(pred.keys())):
            indices.append(pred_sort[i][idx[i]])
        return indices
    
    def get_prob(pred, pred_sort, idx):
        # the idx is the index of pred_sort matrix for each class
        # 1. get the indices in the original pred matrix
        indices = []
        for i in range(len(pred.keys())):
            indices.append(pred_sort[i][idx[i]])
        
        # 2. return the sum of the probabilities
        return sum([pred[key][indices[i]] for i, key in enumerate(pred.keys())])
    
    pred_sorted = []
    for key in ['A', 'B', 'C', 'D', 'E']:
        pred_sorted.append(pred[key].argsort()[-k:][::-1])
        # pred_sorted[key]: top-k indices, sorted by the probability in descending order
    
    largest = [0] * len(pred.keys())
    top_k = []
    boundry = max_heap()
    boundry.push((get_prob(pred, pred_sorted, largest), largest))

    for _ in range(k):
        current = boundry.pop()
        top_k.append(get_indices(pred_sorted, current[1]))

        # expand the current node
        for i in range(len(pred.keys())):
            largest = current[1].copy()
            if largest[i] < len(pred_sorted[i]) - 1:
                largest[i] += 1
                boundry.push((get_prob(pred, pred_sorted, largest), largest))
    return top_k

predictions = []
predictions_bin = []
ground_truths = []
ground_truths_bin = []
for i in range(data_points):
    # split to {A: ...}
    prob_dict = {}
    start = 0
    for key, value in class_dict.items():
        prob_dict[key] = pred[key][i]

    # find the top-10 predictions
    top_k = find_top_k(prob_dict, num_ranks)
    predictions.append(top_k)
    ground_truths.append(list(map(lambda x: int(x[1:]), truth.iloc[i][columns].values.tolist())))
    ground_truths_bin.append([truth.iloc[i][columns].values.tolist()])
    predictions_bin.append([[keys[i] + str(x) for i, x in enumerate(pred)] for pred in top_k])


# save prediction
with open(log_dir + '/top_10_predictions.json', 'w') as fp:
    json.dump(predictions_bin, fp)

with open(log_dir + '/top_10_ground_truths.json', 'w') as fp:
    json.dump(ground_truths_bin, fp)


predictions_smi = []
ground_smi = []
for i in range(len(predictions_bin)):
    predictions_smi.append(['.'.join([id2name[columns[j]][str(x)] for j, x in enumerate(pred) if int(x[1:]) != 0]) for pred in predictions_bin[i]])
    ground_smi.append(['.'.join([id2name[columns[j]][str(x)] for j, x in enumerate(pred) if int(x[1:]) != 0 ]) for pred in ground_truths_bin[i]])

pd.DataFrame({'gt_label': ground_smi, 'pred_label': predictions_smi}).to_csv(log_dir + '/top_10_predictions_smi.csv')