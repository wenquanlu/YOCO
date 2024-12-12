from data.parse_dataset_eval import parse_eval_dataset
import numpy as np
import random

def create_balanced_subset(data, labels):
    baskets = [[] for _ in range(5)]

    for i in range(len(labels)):
        if len(labels[i]) > 10 and len(labels[i]) <= 15:
            baskets[len(labels[i])-11].append(i)
    min_len = np.min([len(_) for _ in baskets])

    test_data = []
    test_labels = []

    i = 0
    for basket in baskets:
        sample_indices = random.sample(basket,min_len)
        for index in sample_indices[:min_len]:
            test_data.append(data[index])
            test_labels.append(labels[index].tolist())
        i += 1
    
    return test_data, test_labels

eval_data, eval_labels = parse_eval_dataset("wider_face_split/wider_face_val_bbx_gt.txt", 15)
test_data, test_labels = create_balanced_subset(eval_data, eval_labels)

f_test = open("generalize_data.py", "w")

f_test.write("gen_data = " + str(test_data))
f_test.write("\n")
f_test.write("gen_label = " + str(test_labels))