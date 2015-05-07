import feature_vector
import sys

def main():
    #args : python trim.py test.ark cv.0.out cv.0.result toSmooth(0|1)
    if (len(sys.argv) < 5):
        print 'command: python trim.py <test_file> <svm_output> <formation_output> <toSmooth>'
        exit()

    d_speechid_index, d_index_phone, d_phone_index, d_phone_alphabet \
        = feature_vector.read_map('../MLDS_HW1_RELEASE_v1/label/train.lab', '../MLDS_HW1_RELEASE_v1/phones/48_idx_chr.map')

    speech_IDs = []
    with open(sys.argv[1], 'r') as fbank:
        lines = fbank.readlines()
        seq_id = 'zzzz'
        for idx, line in enumerate(lines):
            tokens = line.strip().split(' ')
            spch_id = '_'.join(token[0].split('_')[0:2]) # 'a_b_c' --> 'a_b'
            # sequence id changed, store the last example
            if not spch_id == seq_id:
                seq_id = spch_id # 'a_b_c' --> 'a_b'
                speech_IDs.append(seq_id)

    all_y = []
    with open(sys.argv[2], 'r') as f:
        ylines = f.readlines()

    for y in ylines:
        y_list = y.strip('[]\n').split(',')
        y_list = [ele.strip() for ele in y_list]
        trim_y_str = ''
        current_y_idx = 999
        if(sys.argv[4] == '1')
            y_list = smooth(y_list)
        for y_ele in y_list:
            if y_ele != current_y_idx:
                current_y_idx = y_ele
                trim_y_str += d_phone_alphabet[d_index_phone[y_ele]]
        trim_y_str = trim_y_str.strip('L')
        all_y.append(trim_y_str)

    with open(sys.argv[3], 'w') as f:
        print >> f,'id,phone_sequence'
        for idx,y in enumerate(all_y):
            str = speech_IDs[idx] + ',' + all_y[idx]
            print >> f, str

def smooth(y_list):  
    for i in range(1,len(y_list)-1):
        if (y_list[i] != y_list[i-1] & y_list[i] != y_list[i+1]):
            if (y_list[i-2] == y_list[i-1]):
                y_list[i] = y_list[i-1]
    return y_list
            
if __name__ == '__main__':
	main()
