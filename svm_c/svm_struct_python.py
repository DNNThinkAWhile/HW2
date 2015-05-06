import feature_vector

def read_examples(filename):
    """Reads and returns x,y example pairs from a file.

    This reads the examples contained at the file at path filename and
    returns them as a sequence.  Each element of the sequence should
    be an object 'e' where e[0] and e[1] is the pattern (x) and label
    (y) respectively.  Specifically, the intention is that the element
    be a two-element tuple containing an x-y pair."""

    # TODO
    # return a list contained with tuples (data, label)
    print 'reading examples from ', filename

    d_speechid_index, d_index_phone, d_phone_index, d_phone_alphabet \
        = feature_vector.read_map('../MLDS_HW1_RELEASE_v1/label/train.lab', '../MLDS_HW1_RELEASE_v1/phones/48_idx_chr.map')
    examples = []
    DUMMY_STR = 'I am dummy yo'

    with open(filename, 'r') as fbank:
        lines = fbank.readlines()
        print 'readlines done. parsing...'

        # seq_id: like 'faem0_si1392'
        seq_id = DUMMY_STR
        x = []
        y = []

        for idx, line in enumerate(lines):
            tokens = line.strip().split(' ')
            spch_id = tokens[0]
            feat = [float(tok) for tok in tokens[1:]]
            feat.append(1.0)
            # sequence id changed, store the last example
            if not spch_id.split('_')[0:2] == seq_id.split('_'):
            #if not spch_id.startswith(seq_id):
                if seq_id != DUMMY_STR:
                    examples.append((x, y))
                seq_id = '_'.join(spch_id.split('_')[0:2]) # 'a_b_c' --> 'a_b'
                x = []
                y = []

            x.append(feat)
            if spch_id in d_speechid_index:
                y.append(int(d_speechid_index[spch_id]))
            else:
                y.append(-1)


        examples.append((x, y))



    print 'read_examples done.a'
    print 'len:', len(examples)

    return examples
