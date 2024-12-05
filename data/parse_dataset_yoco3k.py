import numpy as np

def parse_dataset_yoco3k(dataset_path="YOCO3k/train/labels/train.txt"):
    threshold = 30
    inspect = 0
    img_count = 0
    #annotations = {}
    data = []
    labels = []
    #curr_file = ""
    curr_label = []
    with open(dataset_path, 'r') as f:
        lines = f.readlines()
        this_len = 0
        counter = 0
        read_file = True
        read_line_num = False
        for line in lines:
            line = line.strip()
            if read_file:
                read_file = False
                read_line_num = True
                img_count += 1
                curr_file = line
                continue
            if read_line_num:
                line_num = int(line)
                if line_num < threshold:
                    inspect += 1
                this_len = line_num
                read_line_num = False
                if this_len > 0:
                    data.append(curr_file)
                if this_len == 0:
                    read_file = True
                    #labels.append(np.array(curr_label))
                    curr_label = []
                continue
            if counter < this_len:
                counter += 1
                anno_numbers = [int(i) for i in line.split()[:4]]
                curr_label.append(anno_numbers)
                if counter == this_len:
                    counter = 0
                    labels.append(np.array(curr_label))
                    curr_label = []
                    read_file = True
                continue
    return data, labels




if __name__ == "__main__":
    data, labels = parse_dataset_yoco3k()
    print(len(data))
    print(len(labels))
    #print(labels[:10])
    agg = [0 for _ in range(11)]
    for label in labels:
        agg[len(label)] += 1
    print(agg)
    