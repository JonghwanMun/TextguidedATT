import argparse
import json

def load_json(file_path) :
    with open(file_path, 'r') as f:
        json_file = json.load(f)
    return json_file

if __name__ == '__main__' :

    parser = argparse.ArgumentParser()
    parser.add_argument('-data_type', dest='data_type', required=True, type=str, help='valtrain|test')
    parser.add_argument('-k', dest='k', required=True, type=int, help='the number of top consensus captions')
    opt = parser.parse_args()
    print 'Options are as follows'
    print opt

    NN_base_path = 'tmp_data/NN_cap_%s%d_cider.json'
    k_base_path = 'tmp_data/%dNN_cap_%s%d_cider.json'
    all_base_path = 'tmp_data/all_cap_%s%d_cider.json'

    if opt.data_type == 'valtrain' :
        num_results = 13
    else :
        num_results = 5

    # merge the result file of consensus captions
    # most consensus caption
    save_json = []
    for i in range(1, num_results+1) :
        ith_consensus_caps = load_json(NN_base_path % (opt.data_type, i))

        for css in ith_consensus_caps :
            save_json.append(css)
        print 'most consensus caption %s%d done' % (opt.data_type, i)

    with open('data/coco/most_consensus_cap_%sall_cider.json'%(opt.data_type), 'w') as f:
        json.dump(save_json, f)

    # top k consensus captions
    save_json = []
    for i in range(1, num_results+1) :
        ith_consensus_caps = load_json(k_base_path % (opt.k, opt.data_type, i))

        for css in ith_consensus_caps :
            save_json.append(css)
        print 'top %d caption %s%d done' % (opt.k, opt.data_type, i)

    with open('data/coco/%dNN_consensus_cap_%sall_cider.json'%(opt.datatype, opt.k), 'w') as f:
        json.dump(save_json, f)

    # all consensus captions
    save_json = []
    for i in range(1, num_results+1) :
        ith_consensus_caps = load_json(all_base_path % (opt.data_type, i))

        for css in ith_consensus_caps :
            save_json.append(css)
        print 'all consensus caption %s%d done' % (opt.data_type, i)

    with open('data/coco/all_consensus_cap_%sall_cider.json'%(opt.data_type), 'w') as f:
        json.dump(save_json, f)
