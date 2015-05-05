import numpy as np
import csv

def read_map(file_label, file_48_idx):
    d_speechid_index = {}
    d_index_phone = {}
    d_phone_index = {}
    d_phone_alphabet = {}
    with open(file_48_idx, 'r') as f:
        for line in f:
            tokens = line.strip().split()
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

def get_psi(fbank_file, d_speechid_index):
    psi = []
    observation = np.zeros(69*48)
    transition = np.zeros(48*48)
    with open(fbank_file, 'r') as f:
	    fbank_data = f.readlines()
    idx = 0
    for line in fbank_data:
        if not line.split(' ')[0].startswith('faem0_si1392'):
            break
        line = line.split('\n')[0]
        phoneme_idx = d_speechid_index[line.split(' ')[0]]
        observation[(phoneme_idx*69): (phoneme_idx*69+69)] = observation[phoneme_idx*69: (phoneme_idx*69+69)] + np.asfarray(line.split(' ')[1:])
        if idx > 0:
            transition[prev_phoneme_idx*48+phoneme_idx] += 1.0
        prev_phoneme_idx = phoneme_idx
        idx = 1
    psi.append(observation)
    psi.append(transition)
    return psi

def outfile_psi(psi):
    with open('psi.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'feature'])
        for n in range(69*48):
            row = ['faem0_si1392_' + str(n), psi[0][n]]
            writer.writerow(row)
        for n in range(48*48):
            row = ['faem0_si1392_' + str(69*48+n), psi[1][n]]
            writer.writerow(row)
    csvfile.close()

def main():
    d_speechid_index, d_index_phone, d_phone_index, d_phone_alphabet \
        = read_map('MLDS_HW1_RELEASE_v1/label/train.lab', 'MLDS_HW1_RELEASE_v1/phones/48_idx_chr.map')
    psi = get_psi('MLDS_HW1_RELEASE_v1/fbank/train.ark', d_speechid_index)
    outfile_psi(psi)
