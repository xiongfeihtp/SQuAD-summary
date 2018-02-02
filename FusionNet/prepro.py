import tensorflow as tf
import random
from tqdm import tqdm
import ujson as json
from collections import Counter
import numpy as np
from keras.models import load_model

def convert_tokens(context, spans, pp1, pp2):
    answer_list = []
    for p1, p2 in zip(pp1, pp2):
        start_idx = spans[p1][0]
        end_idx = spans[p2][1]
        answer_list.append(context[start_idx: end_idx])
    return answer_list

# 解析样本
def vectorize(ex, config, feature_dict):
    # Create extra features vector
    if len(feature_dict) > 0:
        features = np.zeros((len(ex['document']), len(feature_dict)))
    else:
        features = None
    # f_{exact_match}
    if config.use_in_question:
        q_words_cased = {w for w in ex['question']}
        q_words_uncased = {w.lower() for w in ex['question']}
        q_lemma = {w for w in ex['qlemma']}
        for i in range(len(ex['document'])):
            if ex['document'][i] in q_words_cased:
                features[i][feature_dict['in_question']] = 1.0
            if ex['document'][i].lower() in q_words_uncased:
                features[i][feature_dict['in_question_uncased']] = 1.0
            if q_lemma and ex['lemma'][i] in q_lemma:
                features[i][feature_dict['in_question_lemma']] = 1.0
    # f_{token} (POS)
    if config.use_pos:
        for i, w in enumerate(ex['pos']):
            f = 'pos=%s' % w
            if f in feature_dict:
                features[i][feature_dict[f]] = 1.0
    # f_{token} (NER)
    if config.use_ner:
        for i, w in enumerate(ex['ner']):
            f = 'ner=%s' % w
            if f in feature_dict:
                features[i][feature_dict[f]] = 1.0
    # f_{token} (TF)
    if config.use_tf:
        counter = Counter([w.lower() for w in ex['document']])
        l = len(ex['document'])
        for i, w in enumerate(ex['document']):
            features[i][feature_dict['tf']] = counter[w.lower()] * 1.0 / l
    return features


# 导入数据
def load_data(filename):
    # Load JSON lines
    with open(filename) as f:
        examples = [json.loads(line) for line in f]
    # Make case insensitive
    # print('insensitiving')
    # for ex in tqdm(examples):
    #     ex['question'] = [w.lower() for w in ex['question']]
    #     ex['document'] = [w.lower() for w in ex['document']]
    # Skip unparsed (start/end) examples
    examples = [ex for ex in examples if len(ex['answers']) > 0]
    return examples


def build_feature_dict(config, examples):
    """Index features (one hot) from fields in examples and options."""

    def _insert(feature):
        if feature not in feature_dict:
            feature_dict[feature] = len(feature_dict)

    feature_dict = {}
    # Exact match features
    if config.use_in_question:
        _insert('in_question')
        _insert('in_question_uncased')
        _insert('in_question_lemma')
    # Part of speech tag features
    if config.use_pos:
        for ex in examples:
            for w in ex['pos']:
                _insert('pos=%s' % w)
    # Named entity tag features
    if config.use_ner:
        for ex in examples:
            for w in ex['ner']:
                _insert('ner=%s' % w)
    # Term frequency feature
    if config.use_tf:
        _insert('tf')
    return feature_dict


def process_file(raw_data, data_type, word_counter, config, feature_dict):
    print("Generating {} examples...".format(data_type))
    examples = []
    eval_examples = {}
    total = 0
    # 解析指定json文件，统计词频和字母频
    for data_item in tqdm(raw_data):
        context_tokens = data_item['document']
        spans = data_item['offsets']
        context = ' '.join(context_tokens)
        uuid = data_item['id']

        for token in context_tokens:
            word_counter[token] += 1
        ques_tokens = data_item['question']
        total += 1
        for token in ques_tokens:
            word_counter[token] += 1
        y1s, y2s = [], []
        answers = data_item['answers']
        for answer in answers:
            y1s.append(answer[0])
            y2s.append(answer[1])
        answer_texts = convert_tokens(context, spans, y1s, y2s)
        feature = vectorize(data_item, config, feature_dict)
        # 生成examle格式字典
        example = {"feature": feature, "context_tokens": context_tokens,
                   "ques_tokens": ques_tokens,
                   "y1s": y1s, "y2s": y2s, "id": total}
        examples.append(example)
        eval_examples[str(total)] = {"context": context, "spans": spans, "answers": answer_texts, "uuid": uuid}
    random.shuffle(examples)
    print("{} questions in total".format(len(examples)))
    return examples, eval_examples

def get_embedding(counter, data_type, limit=-1, emb_file=None, size=None, vec_size=None):
    print("Generating {} embedding...".format(data_type))
    embedding_dict = {}
    # 滤除低频元素
    filtered_elements = [k for k, v in counter.items() if v > limit]
    if emb_file is not None:
        assert size is not None
        assert vec_size is not None
        with open(emb_file, "r", encoding="utf-8") as fh:
            for line in tqdm(fh, total=size):
                array = line.split()
                # 元素
                word = "".join(array[0:-vec_size])
                # 向量化结果
                vector = list(map(float, array[-vec_size:]))
                # 将符合条件的结果保存到字典
                if word in counter and counter[word] > limit:
                    embedding_dict[word] = vector
        print("{} / {} tokens have corresponding {} embedding vector".format(
            len(embedding_dict), len(filtered_elements), data_type))
    else:
        assert vec_size is not None
        for token in filtered_elements:
            embedding_dict[token] = [np.random.normal(
                scale=0.1) for _ in range(vec_size)]
        print("{} tokens have corresponding embedding vector".format(
            len(filtered_elements)))
    # 设置填充向量和境外元素
    NULL = "--NULL--"
    OOV = "--OOV--"
    token2idx_dict = {token: idx for idx,
                                     token in enumerate(embedding_dict.keys(), 2)}  # idx从2开始，enumerate的特殊用法
    token2idx_dict[NULL] = 0
    token2idx_dict[OOV] = 1
    # 设为零
    embedding_dict[NULL] = [0. for _ in range(vec_size)]
    embedding_dict[OOV] = [0. for _ in range(vec_size)]
    idx2emb_dict = {idx: embedding_dict[token]
                    for token, idx in token2idx_dict.items()}
    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
    return emb_mat, token2idx_dict


def build_features(config, examples, data_type, out_file, word2idx_dict, is_test=False):
    # 设置元素作为序列的最大长度，便于对齐
    para_limit = config.test_para_limit if is_test else config.para_limit
    ques_limit = config.test_ques_limit if is_test else config.ques_limit
    def filter_func(example, is_test=False):
        return len(example["context_tokens"]) > para_limit or len(example["ques_tokens"]) > ques_limit
    # 生成tfrecord file
    print("Processing {} examples...".format(data_type))
    writer = tf.python_io.TFRecordWriter(out_file)
    total = 0
    total_ = 0
    meta = {}
    for example in tqdm(examples):
        total_ += 1
        if filter_func(example, is_test):
            continue
        total += 1
        # 向量
        context_idxs = np.zeros([para_limit], dtype=np.int32)
        # 向量
        ques_idxs = np.zeros([ques_limit], dtype=np.int32)
        # 标签
        y1 = np.zeros([para_limit], dtype=np.float32)
        y2 = np.zeros([para_limit], dtype=np.float32)
        context_feature=np.zeros([para_limit,config.feature_dim],np.float32)
        for i, feature in enumerate(example["feature"]):
            context_feature[i]=feature
        def _get_word(word):
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in word2idx_dict:
                    return word2idx_dict[each]
            return 1
        for i, token in enumerate(example["context_tokens"]):
            context_idxs[i] = _get_word(token)

        for i, token in enumerate(example["ques_tokens"]):
            ques_idxs[i] = _get_word(token)

        start, end = example["y1s"][-1], example["y2s"][-1]
        y1[start], y2[end] = 1.0, 1.0
        # 如何写入序列化向量和矩阵，全部转化为BytesList
        record = tf.train.Example(features=tf.train.Features(feature={
            "context_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[context_idxs.tostring()])),
            "ques_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ques_idxs.tostring()])),
            "y1": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y1.tostring()])),
            "y2": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y2.tostring()])),
            "id": tf.train.Feature(int64_list=tf.train.Int64List(value=[example["id"]])),
            "feature": tf.train.Feature(bytes_list=tf.train.BytesList(value=[context_feature.tostring()]))
        }))
        writer.write(record.SerializeToString())
    print("Build {} / {} instances of features in total".format(total, total_))
    meta["total"] = total
    writer.close()
    return meta

def save(filename, obj, message=None):
    if message is not None:
        print("Saving {}...".format(message))
        with open(filename, "w") as fh:
            json.dump(obj, fh)

def prepro(config):
    # TODO: char part is for cove word embedding file, must be change, here is for glove char
    word_counter= Counter()
    print("loading raw data")
    raw_data_train = load_data(config.train_file)
    raw_data_dev = load_data(config.dev_file)
    raw_data_test = load_data((config.test_file))
    feature_dict = build_feature_dict(config, raw_data_train)
    print(feature_dict)
    train_examples, train_eval = process_file(
        raw_data_train, "train", word_counter, config, feature_dict)
    dev_examples, dev_eval = process_file(
        raw_data_dev, "dev", word_counter, config, feature_dict)
    test_examples, test_eval = process_file(
        raw_data_test, "test", word_counter, config, feature_dict)

    # 选择适当的embedding字向量或者词向量，在更改词向量之前，要提前预处理，生成对应的预处理文件
    word_emb_file = config.fasttext_file if config.fasttext else config.glove_word_file
    char_emb_dim = config.char_dim

    # 生成embedding file和word2idx file，根据还可以根据词频选择词向量分别进行微调，创建不同的word_emb_mat
    print("generation the glove 300 word embedding")
    word_emb_mat, word2idx_dict = get_embedding(
        word_counter, "word", emb_file=word_emb_file, size=config.glove_word_size, vec_size=config.glove_dim)
    #generation the cove 600 word embedding
    print("generation the cove word embedding")
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    cove_model = load_model('Keras_CoVe.h5')
#cove and glove word_emb_mat different word2idx is the same
    char_emb_mat=[]
    for id,raw_vector in enumerate(tqdm(word_emb_mat)):
        if id==0:
            char_emb_mat.append([0. for _ in range(char_emb_dim)])
        elif id==1:
            char_emb_mat.append([0. for _ in range(char_emb_dim)])
        else:
            raw_vector=np.array(raw_vector)
            vector=raw_vector[np.newaxis,np.newaxis, :]
            char_emb_mat.append(np.squeeze(cove_model.predict(vector)).tolist())


    build_features(config, train_examples, "train",config.train_record_file, word2idx_dict)
    dev_meta = build_features(config, dev_examples, "dev",config.dev_record_file, word2idx_dict)
    test_meta = build_features(config, test_examples, "test",config.test_record_file, word2idx_dict, is_test=True)

    save(config.word2idx_dict_file, word2idx_dict, message="word2idx_dict")
    save(config.word_counter_file, dict(word_counter), message="word_counter")
    save(config.word_emb_file, word_emb_mat, message="word embedding")
    save(config.char_emb_file, char_emb_mat, message="char embedding")
    save(config.train_eval_file, train_eval, message="train eval")
    save(config.dev_eval_file, dev_eval, message="dev eval")
    save(config.test_eval_file, test_eval, message="test eval")
    save(config.dev_meta, dev_meta, message="dev meta")
    save(config.test_meta, test_meta, message="test meta")
