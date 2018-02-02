import tensorflow as tf
import random
from tqdm import tqdm
import ujson as json
from collections import Counter
import numpy as np


def process_file(filename, data_type, word_counter, char_counter):
    print("Generating {} examples...".format(data_type))
    examples = []
    total = 0
    # 解析指定json文件，统计词频和字母频
    with open(filename, "r") as fh:
        if data_type == 'train':
            source = json.load(fh)
            for id, article in tqdm(source.items()):
                context = article["context"].replace('#', ' ')
                context_tokens = context.split()
                context_chars = [list(token) for token in context_tokens]
                for token in context_tokens:
                    word_counter[token] += 1
                    for char in token:
                        char_counter[char] += 1
                total += 1
                label = []
                for i in range(1, 13, 2):
                    label.append(int(article['label'][i]))
                # 生成examle格式字典
                example = {"context_tokens": context_tokens, "context_chars": context_chars, "label": label,
                           "id": total}
                examples.append(example)
            random.shuffle(examples)
            print("{} samples in total".format(len(examples)))
        elif data_type == 'eval':
            source = json.load(fh)
            for id, article in tqdm(source.items()):
                context = article["context"].replace('#', ' ')
                context_tokens = context.split()
                context_chars = [list(token) for token in context_tokens]
                for token in context_tokens:
                    word_counter[token] += 1
                    for char in token:
                        char_counter[char] += 1

                total += 1
                example = {"context_tokens": context_tokens, "context_chars": context_chars, "id": total, 'uuid': id}
                examples.append(example)
            random.shuffle(examples)
            print("{} samples in total".format(len(examples)))
    return examples


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
        #对未知元素进行随机初始化
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


def build_features(config, examples, data_type, out_file, word2idx_dict, char2idx_dict, is_test=False):
    # 设置元素作为序列的最大长度，便于对齐
    para_limit = config.test_para_limit if is_test else config.para_limit
    char_limit = config.char_limit
    label_dim = config.label_dim

    def filter_func(example, is_test=False):
        return len(example["context_tokens"]) > para_limit

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
        # 矩阵
        context_char_idxs = np.zeros([para_limit, char_limit], dtype=np.int32)
        # 标签
        label = np.zeros([label_dim], dtype=np.int32)

        def _get_word(word):
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in word2idx_dict:
                    return word2idx_dict[each]
            return 1

        def _get_char(char):
            if char in char2idx_dict:
                return char2idx_dict[char]
            return 1
        for i, token in enumerate(example["context_tokens"]):
            context_idxs[i] = _get_word(token)

        for i, token in enumerate(example["context_chars"]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                context_char_idxs[i, j] = _get_char(char)
        label = example["label"]
        # 如何写入序列化向量和矩阵，全部转化为BytesList
        record = tf.train.Example(features=tf.train.Features(feature={
            "context_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[context_idxs.tostring()])),
            "context_char_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[context_char_idxs.tostring()])),
            "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.tostring()])),
            "id": tf.train.Feature(int64_list=tf.train.Int64List(value=[example["id"]]))
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
    word_counter, char_counter = Counter(), Counter()
    train_examples = process_file(
        config.train_file, "train", word_counter, char_counter)
    test_examples = process_file(
        config.test_file, "test", word_counter, char_counter)

    # 选择适当的embedding字向量或者词向量
    word_emb_file = config.fasttext_file if config.fasttext else config.glove_word_file
    char_emb_file = config.glove_char_file if config.pretrained_char else None
    char_emb_size = config.glove_char_size if config.pretrained_char else None
    char_emb_dim = config.glove_dim if config.pretrained_char else config.char_dim

    # 生成embedding file和word2idx file
    word_emb_mat, word2idx_dict = get_embedding(
        word_counter, "word", emb_file=word_emb_file, size=config.glove_word_size, vec_size=config.glove_dim)
    char_emb_mat, char2idx_dict = get_embedding(
        char_counter, "char", emb_file=char_emb_file, size=char_emb_size, vec_size=char_emb_dim)

    train_meta = build_features(config, train_examples, "train",
                                config.train_record_file, word2idx_dict, char2idx_dict)

    test_meta = build_features(config, test_examples, "test",
                               config.test_record_file, word2idx_dict, char2idx_dict, is_test=True)

    save(config.word_emb_file, word_emb_mat, message="word embedding")
    save(config.char_emb_file, char_emb_mat, message="char embedding")
    save(config.train_eval_file, train_eval, message="train eval")
    save(config.test_eval_file, test_eval, message="test eval")
    save(config.train_meta, train_meta, message="dev meta")
    save(config.test_meta, test_meta, message="test meta")
