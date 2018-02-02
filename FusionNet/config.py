import os
import tensorflow as tf

from prepro import prepro
from main import train, test

flags = tf.flags
# TODO: This part is a little confusing char->cove
home = os.path.expanduser("~")
train_file = os.path.join(home, "data", "squad", "train-v1.1-processed-corenlp.txt")
dev_file = os.path.join(home, "data", "squad", "dev-v1.1-processed-corenlp.txt")
test_file = os.path.join(home, "data", "squad", "dev-v1.1-processed-corenlp.txt")
glove_word_file = os.path.join(home, "data", "glove", "glove.840B.300d.txt")

flags.DEFINE_string("run_id","0","RUN ID[0]")
flags.DEFINE_string("model_name","basic","model name")

target_dir = "data"
log_dir = os.path.join(flags.FLAGS.model_name,flags.FLAGS.run_id,"event")
save_dir = os.path.join(flags.FLAGS.model_name,flags.FLAGS.run_id,"save")
answer_dir = os.path.join(flags.FLAGS.model_name,flags.FLAGS.run_id,"answer")

train_record_file = os.path.join(target_dir, "train.tfrecords")
dev_record_file = os.path.join(target_dir, "dev.tfrecords")
test_record_file = os.path.join(target_dir, "test.tfrecords")


word_emb_file = os.path.join(target_dir, "word_emb.json")
char_emb_file = os.path.join(target_dir, "char_emb.json")
word2idx_dict_file =os.path.join(target_dir, "word2idx_dict")
word_counter_file=os.path.join(target_dir,"word_counter")
flags.DEFINE_string("word2idx_dict_file", word2idx_dict_file, "Out file for char embedding")
flags.DEFINE_string("word_counter_file", word_counter_file, "Out file for char embedding")

train_eval = os.path.join(target_dir, "train_eval.json")
dev_eval = os.path.join(target_dir, "dev_eval.json")
test_eval = os.path.join(target_dir, "test_eval.json")
dev_meta = os.path.join(target_dir, "dev_meta.json")
test_meta = os.path.join(target_dir, "test_meta.json")
answer_file = os.path.join(answer_dir, "answer.json")

if not os.path.exists(target_dir):
    os.makedirs(target_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if not os.path.exists(answer_dir):
    os.makedirs(answer_dir)


flags.DEFINE_string("mode", "train", "Running mode train/debug/test")
flags.DEFINE_string("target_dir", target_dir, "Target directory for out data")
flags.DEFINE_string("log_dir", log_dir, "Directory for tf event")
flags.DEFINE_string("save_dir", save_dir, "Directory for saving model")
flags.DEFINE_string("train_file", train_file, "Train source file")
flags.DEFINE_string("dev_file", dev_file, "Dev source file")
flags.DEFINE_string("test_file", test_file, "Test source file")
flags.DEFINE_string("glove_word_file", glove_word_file, "Glove word embedding source file")
flags.DEFINE_string("load_path",None,"retrain_path")
flags.DEFINE_integer("load_step",0,"retrain globel step")

flags.DEFINE_string("train_record_file", train_record_file, "Out file for train data")
flags.DEFINE_string("dev_record_file", dev_record_file, "Out file for dev data")
flags.DEFINE_string("test_record_file", test_record_file, "Out file for test data")
flags.DEFINE_string("word_emb_file", word_emb_file, "Out file for word embedding")
flags.DEFINE_string("char_emb_file", char_emb_file, "Out file for char embedding")
flags.DEFINE_string("train_eval_file", train_eval, "Out file for train eval")
flags.DEFINE_string("dev_eval_file", dev_eval, "Out file for dev eval")
flags.DEFINE_string("test_eval_file", test_eval, "Out file for test eval")
flags.DEFINE_string("dev_meta", dev_meta, "Out file for dev meta")
flags.DEFINE_string("test_meta", test_meta, "Out file for test meta")
flags.DEFINE_string("answer_file", answer_file, "Out file for answer")


flags.DEFINE_boolean("use_in_question", True, "exact_match")
flags.DEFINE_boolean("use_pos", True, "pos feature")
flags.DEFINE_boolean("use_ner", True, "ner feature")
flags.DEFINE_boolean("use_tf", True, "ner feature")


flags.DEFINE_integer("cove_word_size", int(2.2e6), "Corpus size for cove")
flags.DEFINE_integer("glove_word_size", int(2.2e6), "Corpus size for Glove")

flags.DEFINE_integer("glove_dim", 300, "Embedding dimension for Glove")
flags.DEFINE_integer("char_dim", 600, "Embedding dimension for char")

flags.DEFINE_integer("feature_dim",62,"context new feature dim")
flags.DEFINE_integer("para_limit", 400, "Limit length for paragraph")
flags.DEFINE_integer("ques_limit", 50, "Limit length for question")
flags.DEFINE_integer("test_para_limit", 1000, "Limit length for paragraph in test file")
flags.DEFINE_integer("test_ques_limit", 100, "Limit length for question in test file")

flags.DEFINE_integer("word_count_limit", -1, "Min count for word")

flags.DEFINE_integer("capacity", 15000, "Batch size of dataset shuffle")
flags.DEFINE_integer("num_threads", 4, "Number of threads in input pipeline")
flags.DEFINE_boolean("use_cudnn", True, "Whether to use cudnn rnn (should be False for CPU)")
flags.DEFINE_boolean("is_bucket", False, "build bucket batch iterator or not")
flags.DEFINE_integer("bucket_range", [40, 401, 40], "the range of bucket")

flags.DEFINE_integer("batch_size", 4, "Batch size")
flags.DEFINE_integer("num_steps", 60000, "Number of steps")
flags.DEFINE_integer("checkpoint", 1000, "checkpoint to save and evaluate the model")
flags.DEFINE_integer("period", 100, "period to save batch loss")
flags.DEFINE_integer("val_num_batches", 150, "Number of batches to evaluate the model")
flags.DEFINE_float("init_lr", 0.5, "Initial learning rate for Adadelta")

flags.DEFINE_float("keep_prob", 0.4, "Dropout keep prob in rnn, 0.4 in paper")
flags.DEFINE_float("ptr_keep_prob", 0.7, "Dropout keep prob for pointer network")

flags.DEFINE_float("grad_clip", 5.0, "Global Norm gradient clipping rate")
#flags.DEFINE_integer("hidden", 75, "Hidden size")
#flags.DEFINE_integer("char_hidden", 100, "GRU dimention for char")
flags.DEFINE_integer("patience", 3, "Patience for learning rate decay")
flags.DEFINE_integer("max_to_keep",None,"max numbers of saved model")

fasttext_file = os.path.join(home, "data", "fasttext", "wiki-news-300d-1M.vec")
flags.DEFINE_string("fasttext_file", fasttext_file, "Fasttext word embedding source file")
flags.DEFINE_boolean("fasttext", False, "Whether to use fasttext")

flags.DEFINE_integer("cove_dim",600,"cove_dim")
flags.DEFINE_integer("reading_rep_dim",50,"reading_rep_dim")
flags.DEFINE_integer("final_ques_under_dim",50,"final_ques_under_dim")
flags.DEFINE_integer("fully_fused_para_dim",50,"fully_fused_para_dim")
flags.DEFINE_integer("sl_att_dim",13,"sl_att_dim")
flags.DEFINE_integer("sh_att_dim",12,"sh_att_dim")
flags.DEFINE_integer("su_att_dim",11,"su_att_dim")
flags.DEFINE_integer("selfboost_att_dim",10,"selfboost_att_dim")
flags.DEFINE_integer("selfboost_rep_dim",8,"selfboost_rep_dim")
def main(_):
    config = flags.FLAGS
    if config.mode == "train":
        train(config)
    elif config.mode == "prepro":
        prepro(config)
    elif config.mode == "debug":
        config.num_steps = 2
        config.val_num_batches = 1
        config.checkpoint = 1
        config.period = 1
        train(config)
    elif config.mode == "test":
        if config.use_cudnn:
            print("Warning: Due to a known bug in Tensorlfow, the parameters of CudnnGRU may not be properly restored.")
        test(config)
    else:
        print("Unknown mode")
        exit(0)


if __name__ == "__main__":
    tf.app.run()
