from tqdm import tqdm
from collections import defaultdict
import json
import tensorflow as tf
import os
import gzip
import pickle
flags = tf.flags
flags.DEFINE_string("file_pattern", './ensemble_item/*', "Out file for train data")

#target_dir = "data"
#test_record_file = os.path.join(target_dir, "test.tfrecords")
flags.DEFINE_string("test_eval_file", './data/test_eval.json', "Out file for test data")
flags.DEFINE_string("save_dir", './ensemble_result.json', "Directory for saving result")
#from util import get_record_parser



def convert_tokens(context, spans, p1, p2):
   # print(p1,p2,context,spans)
    try:
        start_idx = spans[p1][0]
        end_idx = spans[p2][1]
    except Exception as e:
        print(context)
        print(spans)
        print(p1,p2)
    return context[start_idx: end_idx]


def ensemble(config, model_list):
    # raw_datai
    with open(config.test_eval_file, "r") as fh:
        raw_data = json.load(fh)

    # # tf-record data
    # dataset = tf.data.TFRecordDataset(config.test_record_file).map(
    #     get_record_parser(config, is_test=True))

    e_list = []
    for path in model_list:
        with gzip.open(path, 'r') as fh:
            e = pickle.load(fh)
            e_list.append(e)
    out = {}
    for idx, item in tqdm(raw_data.items()):
        context = item['context']
        spans = item['spans']
        yp_list = [e[idx]['yp1'] for e in e_list]
        yp2_list = [e[idx]['yp2'] for e in e_list]
        answer = ensemble_cal(context, spans, yp_list, yp2_list)
        uuid = item['uuid']
        out[uuid] = answer
    with open(config.save_dir, 'w') as fh:
        json.dump(out, fh)

def get_span_score_pairs_me(ypi,yp2i):
    span_score_pairs=[]
    for j in range(len(ypi)):
        for k in range(j+1,len(yp2i)):
            span=(j,k)
            score=ypi[j]*yp2i[k]
            span_score_pairs.append((span,score))
    return span_score_pairs

def get_span_score_pairs(ypi, yp2i):
    span_score_pairs = []
    for f, (ypif, yp2if) in enumerate(zip(ypi, yp2i)):
        print(ypif,yp2if)
        for j in range(len(ypif)):
            for k in range(j, len(yp2if)):
                span = ((f, j), (f, k+1))
                score = ypif[j] * yp2if[k]
                span_score_pairs.append((span, score))
    print(span_score_pairs)
    return span_score_pairs

def ensemble_cal(context, wordss, y1_list, y2_list):
    d = defaultdict(lambda: 0.0)
    # 概率计算
    for y1, y2 in zip(y1_list, y2_list):
        for span, score in get_span_score_pairs_me(y1, y2):
            d[span] += score
    span = max(d.items(), key=lambda pair: pair[1])[0]
    # 选最大的
    phrase = convert_tokens(context, wordss, span[0], span[1])
  #  print(span)
  #  print(context)
  #  print(wordss)
  #  print(phrase)
    return phrase

def main():
    config = flags.FLAGS
    match = tf.gfile.Glob(config.file_pattern)
    print("Found %d model.", len(match))
    print(match)
    ensemble(config,match)
    print("finish enseble")
if __name__ == "__main__":
    main()
