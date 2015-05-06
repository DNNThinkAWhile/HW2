import feature_vector
import sys

def main():
    #args : python trim.py test.ark cv.0.out cv.0.result
    if (len(sys.argv) < 4):
        print 'command: python trim.py <test_file> <svm_output> <formation_output>'
        exit()

    d_speechid_index, d_index_phone, d_phone_index, d_phone_alphabet \
        = feature_vector.read_map('../MLDS_HW1_RELEASE_v1/label/train.lab', '../MLDS_HW1_RELEASE_v1/phones/48_idx_chr.map')

    speech_IDs = []
    with open(sys.argv[1], 'r') as fbank:
        lines = fbank.readlines()
        seq_id = 'zzzz'
        for idx, line in enumerate(lines):
            tokens = line.strip().split(' ')
            spch_id = tokens[0]
            # sequence id changed, store the last example
            if spch_id.split('_')[0:2] != seq_id.split('_'):
                seq_id = '_'.join(spch_id.split('_')[0:2]) # 'a_b_c' --> 'a_b'
                speech_IDs.append(seq_id)

    all_y = []
    with open(sys.argv[2], 'r') as f:
        ylines = f.readlines()

    for y in ylines:
        y_list = y.strip('[]\n').split(',')
        trim_y_str = ''
        current_y_idx = 999
        for y_ele in y_list:
            if y_ele != current_y_idx:
                current_y_idx = y_ele
                trim_y_str += d_phone_alphabet[d_index_phone[y_ele.strip()]]
        trim_y_str = trim_y_str.strip('L')
        all_y.append(trim_y_str)

    with open(sys.argv[3], 'w') as f:
        print >> f,'id,phone_sequence'
        for idx,y in enumerate(all_y):
            str = speech_IDs[idx] + ',' + all_y[idx]
            print >> f, str

if __name__ == '__main__':
	main()
