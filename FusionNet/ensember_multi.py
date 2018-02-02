from tqdm import tqdm
from collections import defaultdict
import json
import tensorflow as tf
import os
import gzip
import pickle
flags = tf.flags
import numpy as np
from multiprocessing import Pool as ProcessPool
flags.DEFINE_string("file_pattern", './ensemble_item/*', "Out file for train data")

#target_dir = "data"
#test_record_file = os.path.join(target_dir, "test.tfrecords")
flags.DEFINE_string("test_eval_file", './data/test_eval.json', "Out file for test data")
flags.DEFINE_string("save_dir", './ensemble_result.json', "Directory for saving result")
#from util import get_record_parser

e_list=None
f_answer=open('./answer_file.txt','w')
f_write=None
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

def func(pair):
    global f_write
    item=pair[1]
    idx=pair[0]
    context = item['context']
    spans = item['spans']
    yp_list = [e[idx]['yp1'] for e in e_list]
    yp2_list = [e[idx]['yp2'] for e in e_list]
    answer = ensemble_cal(context, spans, yp_list, yp2_list)
    uuid = item['uuid']
    outline="&".join([uuid,answer])+"\n"
    f_write.write(outline)
    f_write.flush()
def init(ensemble_list,filename):
    global e_list
    global f_write
    e_list=ensemble_list
    f_write=filename
def ensemble(config, model_list):
    global f_answer
    with open(config.test_eval_file, "r") as fh:
        raw_data = json.load(fh)
    # # tf-record data
    # dataset = tf.data.TFRecordDataset(config.test_record_file).map(
    #     get_record_parser(config, is_test=True))
    e_list_med = []
    print('single model accuracy and f1:')
    for path in model_list:
        with gzip.open(path, 'r') as fh:
            e = pickle.load(fh)
            print(path,"em:{}".format(e['exact_math']),"f1:{}".format(e['f1']))
            e_list_med.append(e)
    #training data_precessing
    num_workers = 24
    print('write training data number_workers:{}'.format(num_workers))
    # 多进程编程思路
    workers = ProcessPool(num_workers,initializer=init,initargs=(e_list_med,f_answer,))
    # tqdm对象的用法
    with tqdm(total=len(raw_data.items())) as pbar:
        for pairs in tqdm(workers.imap_unordered(func, raw_data.items())):
            pbar.update()
    # for idx, item in tqdm(raw_data.items()):
    #     func(item,e_list,idx)
    f_answer.close()
    out={}
    with open('./answer_file.txt','r') as f:
        lines=f.readlines()
        for line in lines:
            com=line.strip().split('&')
            try:
                out[com[0]]=com[1]
            except Exception as e:
                print(com)
    with open(config.save_dir, 'w') as fh:
        json.dump(out, fh)

def get_span_score_pairs_np(ypi,yp2i):
    ypi_np=np.array(ypi)
    yp2i_np=np.array(yp2i)
    outer=np.dot(ypi_np[:,np.newaxis],yp2i_np[np.newaxis,:])
    s_diag=outer-np.tril(outer,-1)
    outer=np.tril(s_diag,15)
    yp1=np.max(outer,axis=1)
    yp2=np.max(outer,axis=0)
    return yp1,yp2

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
    d1 = defaultdict(lambda: 0.0)
    d2= defaultdict(lambda: 0.0)
    # 概率计算
    for y1, y2 in zip(y1_list, y2_list):
        yp1,yp2=get_span_score_pairs_np(y1,y2)
        for i, (p1,p2) in enumerate(zip(yp1,yp2)):
            d1[i] += p1
            d2[i]+=p2
            span = (max(d1.items(), key=lambda pair: pair[1])[0],max(d2.items(),key=lambda pair: pair[1])[0])
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
