# from data.parse_dataset_eval import parse_eval_dataset
# import numpy as np
# import random

# def create_balanced_subset(max_count, data, labels):
#     baskets = [[] for _ in range(max_count)]

#     for i in range(len(labels)):
#         if len(labels[i]) > 0 and len(labels[i]) <= max_count:
#             baskets[len(labels[i])-1].append(i)
#     min_len = np.min([len(_) for _ in baskets])
    
#     val_data = []
#     val_labels = []

#     test_data = []
#     test_labels = []

#     i = 0
#     for basket in baskets:
#         sample_indices = random.sample(basket,min_len)
#         for index in sample_indices[:int(min_len/2)]:
#             test_data.append(data[index])
#             test_labels.append(labels[index].tolist())
#         for index in sample_indices[int(min_len/2):]:
#             val_data.append(data[index])
#             val_labels.append(labels[index].tolist())
#         i += 1
    
#     return test_data, test_labels, val_data, val_labels

# eval_data, eval_labels = parse_eval_dataset("wider_face_split/wider_face_val_bbx_gt.txt", 10)
# test_data, test_labels, val_data, val_labels = create_balanced_subset(10, eval_data, eval_labels)

# print(len(test_data), len(test_labels), len(val_data), len(val_labels))

# f_test = open("test_data.py", "w")

# f_test.write("test_data = " + str(test_data))
# f_test.write("\n")
# f_test.write("test_label = " + str(test_labels))

# f_val = open("val_data.py", "w")
# f_val.write("val_data = " + str(val_data))
# f_val.write("\n")
# f_val.write("val_label = " + str(val_labels))

from val_data import val_data, val_label
from test_data import test_data, test_label
print(len(val_data))
print(len(val_label))
print(len(test_data))
print(len(test_label))
