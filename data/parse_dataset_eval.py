import numpy as np

def parse_eval_dataset(dataset_path="wider_face_split/wider_face_val_bbx_gt.txt", max_count=1):
    inspect = 0
    img_count = 0
    #annotations = {}
    data = []
    labels = []
    curr_label = []
    f_write = open("train.txt", "w")
    with open(dataset_path, 'r') as f:
        lines = f.readlines()
        this_len = 0
        counter = 0
        read_file = True
        read_line_num = False
        #print(lines[:5])
        curr_file = ""
        for line in lines:
            line = line.strip()
            #print(line, "#####")
            if read_file:
                read_file = False
                read_line_num = True
                #print(line == "0--Parade/0_Parade_Parade_0_452.jpg")
                #print(line, "!!!")
                #print("file:", line)
                curr_file = line
                img_count += 1
                continue
            if read_line_num:
                line_num = int(line)

                this_len = line_num
                if line_num <= max_count and line_num > 0:
                    data.append(curr_file)
                #print(this_len)
                read_line_num = False
                continue
            #if this_len == 0:
            #    #annotations[curr_file] = []
            #    print(line, "thislen0")
            #    labels.append(np.array(curr_label))
            #    curr_label = []
            #    read_file = True
            #    continue
            if this_len == 0:
                read_file = True
                #labels.append(np.array(curr_label))
                curr_label = []
                continue
            if counter < this_len:
                counter += 1
                anno_numbers = [int(i) for i in line.split()[:4]]
                curr_label.append(anno_numbers)
                #annotations.setdefault(curr_file, []).append(anno_numbers)
                if counter == this_len:
                    counter = 0
                    if this_len <= max_count and this_len > 0:
                        labels.append(np.array(curr_label))
                    curr_label = []
                    read_file = True
                continue
    return data, labels

#data, label = parse_eval_dataset()
#print(len(data))
#print(len(label))