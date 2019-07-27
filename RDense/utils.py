import numpy as np

STRUCTURES = 5
# -*- coding: utf-8 -*-
def process_clamp(clamp_file):   
    data = clamp_file.readline()
    if not data:
        return None
    line = data.strip().split()
    score = float(line[0])
    seq = line[1]
    return (seq, score)

def process_rnacontext(rnacontext_file):
    data = rnacontext_file.readline()
    if not data:
        return None
    seq_line = data.strip()
    assert (seq_line[0] == '>')
    seq = seq_line[1:]
    matrix = list()
    for structure_index in range(STRUCTURES):
        structure_line = rnacontext_file.readline().strip()
        matrix_line = [float(elem) for elem in structure_line.split()]
        matrix.append(matrix_line)
    return (seq, matrix)

def read_sequence_only(sequences_path, structures_path, max_seq_len):
    with open(sequences_path, 'r') as sequences, open(structures_path, 'r') as structures:
        data = list()
        lengths = list()
        labels = list()
        counter = 0
        while True:
            counter += 1
            seq_data = process_clamp(sequences)
            structure_data = process_rnacontext(structures)
            if not seq_data or not structure_data:
                return np.array(data), np.array(lengths),np.array(labels), counter-1
            #print ("Line", counter)
            #print (seq_data)
            #print (structure_data)
            labels.append(seq_data[1])
            seq_matrix = list()
            for base in structure_data[0]:
                if base == 'A':
                    base_encoding = [1, 0, 0, 0]
                elif base == 'C':
                    base_encoding = [0, 1, 0, 0]
                elif base == 'G':
                    base_encoding = [0, 0, 1, 0]
                elif base == 'U':
                    base_encoding = [0, 0, 0, 1]
                else:
                    raise ValueError
                seq_matrix.append(base_encoding)
            seq_matrix = np.array(seq_matrix)
            struct_matrix = np.transpose(np.array(structure_data[1]))
            ver_diff = STRUCTURES - seq_matrix.shape[1]
            assert (ver_diff >= 0)
            if ver_diff > 0:
                padding_columns = np.zeros((seq_matrix.shape[0], ver_diff))
                seq_matrix = np.concatenate((seq_matrix, padding_columns), axis=1)
            # each RNA seq should be of MAX_SEQ_LEN
            curr_seq_len = seq_matrix.shape[0]
            lengths.append(curr_seq_len)
            padd_len = max_seq_len - curr_seq_len
            assert (padd_len  >= 0)
            if padd_len > 0:
                padding_matrix = np.zeros((padd_len, STRUCTURES))
                seq_matrix = np.concatenate((seq_matrix, padding_matrix), axis=0)
                struct_matrix = np.concatenate((struct_matrix, padding_matrix), axis=0)
            base_matrix=seq_matrix
            #print (base_matrix.shape)
            data.append(base_matrix)
        assert(False)

def read_combined_data(sequences_path, structures_path, max_seq_len):
    with open(sequences_path, 'r') as sequences, open(structures_path, 'r') as structures:
        data = list()
        lengths = list()
        # labels = list()
        counter = 0
        while True:
            counter += 1
            seq_data = process_clamp(sequences)
            structure_data = process_rnacontext(structures)
            if not seq_data or not structure_data:
                return np.array(data), np.array(lengths), counter-1
            seq_matrix = list()
            for base in structure_data[0]:
                if base == 'A':
                    base_encoding = [1, 0, 0, 0]
                elif base == 'C':
                    base_encoding = [0, 1, 0, 0]
                elif base == 'G':
                    base_encoding = [0, 0, 1, 0]
                elif base == 'U':
                    base_encoding = [0, 0, 0, 1]
                else:
                    raise ValueError
                seq_matrix.append(base_encoding)
            seq_matrix = np.array(seq_matrix)
            struct_matrix = np.transpose(np.array(structure_data[1]))
            ver_diff = STRUCTURES - seq_matrix.shape[1]
            assert (ver_diff >= 0)
            if ver_diff > 0:
                padding_columns = np.zeros((seq_matrix.shape[0], ver_diff))
                seq_matrix = np.concatenate((seq_matrix, padding_columns), axis=1)
            curr_seq_len = seq_matrix.shape[0]
            lengths.append(curr_seq_len)
            padd_len = max_seq_len - curr_seq_len
            assert (padd_len  >= 0)
            if padd_len > 0:
                padding_matrix = np.zeros((padd_len, STRUCTURES))
                struct_matrix = np.concatenate((struct_matrix, padding_matrix), axis=0)
            base_matrix=struct_matrix
            #print (base_matrix.shape)
            data.append(base_matrix)
        assert(False)
    # return data, lengths, labels


