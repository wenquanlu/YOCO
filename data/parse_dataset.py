import numpy as np

def parse_dataset(dataset_path="YOCO3k/train/labels/train.txt"):
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
        #print(lines[:5])
        for line in lines:
            line = line.strip()
            #print(line, "#####")
            if read_file:
                read_file = False
                read_line_num = True
                #print(line == "0--Parade/0_Parade_Parade_0_452.jpg")
                #print(line, "!!!")
                #curr_file = line
                #print("file:", line)
                data.append(line)
                img_count += 1
                continue
            if read_line_num:
                line_num = int(line)
                if line_num < threshold:
                    inspect += 1
                this_len = line_num
                #print(this_len)
                read_line_num = False
                if this_len == 0:
                    read_file = True
                    labels.append(np.array(curr_label))
                    curr_label = []
                continue
            #if this_len == 0:
            #    #annotations[curr_file] = []
            #    print(line, "thislen0")
            #    labels.append(np.array(curr_label))
            #    curr_label = []
            #    read_file = True
            #    continue
            if counter < this_len:
                counter += 1
                anno_numbers = [int(i) for i in line.split()[:4]]
                curr_label.append(anno_numbers)
                #annotations.setdefault(curr_file, []).append(anno_numbers)
                if counter == this_len:
                    counter = 0
                    labels.append(np.array(curr_label))
                    curr_label = []
                    read_file = True
                continue
    return data, labels




if __name__ == "__main__":
    data, labels = parse_dataset()
    print(len(data))
    print(len(labels))
    print(labels[:10])
