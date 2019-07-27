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
    for structure_index in range(4):
        structure_line = rnacontext_file.readline().strip()
        matrix_line = [float(elem) for elem in structure_line.split()]
        matrix.append(matrix_line)
    return (seq, matrix)

def read_features(feature_path):
    with open(feature_path, 'r') as features:
        data = list()
        lengths = list()
        # labels = list()
        counter = 0
        while True:
            counter += 1

            structure_data = process_rnacontext(features)
            if not structure_data:
                return np.array(data), np.array(lengths), counter-1
            struct_matrix = np.transpose(np.array(structure_data[1]))
            curr_seq_len = struct_matrix.shape[0]
            lengths.append(curr_seq_len)
            base_matrix=struct_matrix

            data.append(base_matrix)            
        assert(False)
