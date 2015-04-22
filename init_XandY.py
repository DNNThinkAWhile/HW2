import numpy as np
import csv

FBANK_DIM = 69
PHONE_NUM = 48
PSI_SIZE = 5616           #69*48 + 48*48
OBSERVATION_SIZE = 3312   #69*48

def read_map(file_label, file_48_idx):
    d_speechid_index = {}
    d_index_phone = {}
    d_phone_index = {}
    d_phone_alphabet = {}
    with open(file_48_idx, 'r') as f:
        for line in f:
            tokens = line.strip('\n').split()            
            # map phonemes(sil, aa,..) to alphabets(a, b, c,..)
            d_phone_alphabet[tokens[0]] = tokens[2]
            # map phonemes(sil, aa,..) to their indices(int), 0:47
            d_phone_index[tokens[0]] = tokens[1]
            # map index(int), 0:47 to phonemes(sil, aa,..)
            d_index_phone[tokens[1]] = tokens[0]

    # map speech_id(like "maeb0_si1411_1") to their phonemes' indices, 1943->48
    with open(file_label, 'r') as f:
        for line in f:
            tokens = line.strip().split(',')
            d_speechid_index[tokens[0]] = d_phone_index[tokens[1]]
    return d_speechid_index, d_index_phone, d_phone_index, d_phone_alphabet

def get_psi(fbank_file, d_speechid_index, d_index_phone, d_phone_alphabet):
    all_psi = []
    all_x = []
    all_y =[]
    with open(fbank_file, 'r') as f:
	    fbank_data = f.readlines()
    idx = 0
    set_speechID = ''
    for line in fbank_data:
        line_token = line.split()
        speechID = line_token[0].rstrip('1234567890')
        if set_speechID != speechID:
            set_psi = np.zeros(PSI_SIZE)
            set_x =[]
            set_y =''
            all_psi.append(set_psi)
            all_x.append(set_x)
            all_y.append(set_y)
            set_speechID = speechID
            idx = 0
        phoneme_idx = (int)(d_speechid_index[line_token[0]])
        all_psi[-1][(phoneme_idx*69): (phoneme_idx*69+69)] = all_psi[-1][phoneme_idx*69: (phoneme_idx*69+69)] + np.asfarray(line_token[1:])
        all_x[-1].append(np.asfarray(line_token[1:]))
        if idx > 0:
            all_psi[-1][OBSERVATION_SIZE + prev_phoneme_idx*48 + phoneme_idx] += 1.0
        prev_phoneme_idx = phoneme_idx
        idx = 1
        all_y[-1] = all_y[-1] + d_phone_alphabet[ d_index_phone[ d_speechid_index[line_token[0]] ] ]
    return all_x, all_y, all_psi
    

d_speechid_index, d_index_phone, d_phone_index, d_phone_alphabet \
    = read_map('MLDS_HW1_RELEASE_v1/label/train.lab', 'MLDS_HW1_RELEASE_v1/phones/48_idx_chr.map')
x,y,psi = get_psi('MLDS_HW1_RELEASE_v1/fbank/train.ark', d_speechid_index, d_index_phone, d_phone_alphabet)
print len(x[0]), len(x[0][0])
print y[0]
print len(phi[0])
