import tensorflow as tf
from func import *
import logging
logging.getLogger().setLevel("DEBUG")
birnn = tf.nn.bidirectional_dynamic_rnn

# 学习输入文本的框架，完成前面没完成的attention网络
# change char_mat to cove_mat 600dim
class Model(object):
    def __init__(self, config, batch, word_mat=None, char_mat=None, trainable=True, opt=True):
        self.config = config
        self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                           initializer=tf.constant_initializer(0), trainable=False)
        # 这个框架非常好理解，学习数据处理的流程
        self.c, self.q, self.y1, self.y2, self.qa_id, self.context_feature = batch.get_next()
        self.is_train = tf.get_variable(
            "is_train", shape=[], dtype=tf.bool, trainable=False)

        self.word_mat = tf.get_variable("word_mat", initializer=tf.constant(
            word_mat, dtype=tf.float32), trainable=False)
        # tf.add_to_collection(tf.GraphKeys.TRAINABLE_VARIABLES,self.word_mat)
        self.char_mat = tf.get_variable(
            "cove_mat", initializer=tf.constant(char_mat, dtype=tf.float32), trainable=False)
        N = config.batch_size

        self.c_mask = tf.cast(self.c, tf.bool)
        self.q_mask = tf.cast(self.q, tf.bool)
        self.c_len = tf.reduce_sum(tf.cast(self.c_mask, tf.int32), axis=1)
        self.q_len = tf.reduce_sum(tf.cast(self.q_mask, tf.int32), axis=1)
        # 优化对齐
        if opt:
            N = config.batch_size
            self.c_maxlen = tf.reduce_max(self.c_len)
            self.q_maxlen = tf.reduce_max(self.q_len)
            self.c = tf.slice(self.c, [0, 0], [N, self.c_maxlen])
            self.q = tf.slice(self.q, [0, 0], [N, self.q_maxlen])
            self.c_mask = tf.slice(self.c_mask, [0, 0], [N, self.c_maxlen])
            self.q_mask = tf.slice(self.q_mask, [0, 0], [N, self.q_maxlen])
            self.y1 = tf.slice(self.y1, [0, 0], [N, self.c_maxlen])
            self.y2 = tf.slice(self.y2, [0, 0], [N, self.c_maxlen])
            self.context_feature = tf.slice(self.context_feature, [0, 0, 0], [N, self.c_maxlen, config.feature_dim])
        else:
            self.c_maxlen, self.q_maxlen = config.para_limit, config.ques_limit
        self.ready()
        if trainable:
            self.lr = tf.get_variable(
                "lr", shape=[], dtype=tf.float32, trainable=False)
            self.opt = tf.train.AdadeltaOptimizer(
                learning_rate=self.lr, epsilon=1e-6)
            grads = self.opt.compute_gradients(self.loss)
            gradients, variables = zip(*grads)
            capped_grads, _ = tf.clip_by_global_norm(
                gradients, config.grad_clip)
            self.train_op = self.opt.apply_gradients(
                zip(capped_grads, variables), global_step=self.global_step)

    def ready(self):
        config = self.config
        N, PL, QL, d_gl, dc = config.batch_size, self.c_maxlen, self.q_maxlen, config.glove_dim, config.char_dim
        gru = cudnn_gru if config.use_cudnn else native_gru
        logging.info("feature embedding")
        # glove
        c_emb = tf.reshape(tf.nn.embedding_lookup(
            self.word_mat, self.c), [N, PL, d_gl])
        q_emb = tf.reshape(tf.nn.embedding_lookup(
            self.word_mat, self.q), [N, QL, d_gl])
        # cove
        ch_emb = tf.reshape(tf.nn.embedding_lookup(
            self.char_mat, self.c), [N, PL, dc])
        qh_emb = tf.reshape(tf.nn.embedding_lookup(
            self.char_mat, self.q), [N, QL, dc])

        # new_feature
        c_f = tf.reshape(tf.cast(self.context_feature, tf.float32), [N, PL, config.feature_dim])

        logging.info("Word level infusion")
        # fused_a = fuse(para_glove, ques_glove, attention_dim, 'test')
        # high_level  dropout fusion with gate attention
        para_q_fused_glove = word_fusion_vecotr(c_emb, q_emb,
                                                self.q_mask, config.keep_prob, self.is_train, scope="word_fusion")
        # para_q_fused_glove = word_fusion(c_emb, q_emb,
        #                                  self.c_mask, self.q_mask)
        # low_level
        para_w_rep = tf.concat([c_emb, ch_emb, c_f], axis=2)

        # low_level
        ques_w_rep = tf.concat([q_emb, qh_emb], axis=2)

        # enhanced input vector for context
        para_enhanced_rep = tf.concat([para_w_rep, para_q_fused_glove], axis=2)
        # ---------------------reading
        logging.info("Building Reading section")
        # change LSTM to GRU
        with tf.variable_scope("Reading"):
            # hQh
            f_read_q_low = tf.contrib.rnn.LSTMCell(config.reading_rep_dim // 2)  # in paper 125
            b_read_q_low = tf.contrib.rnn.LSTMCell(config.reading_rep_dim // 2)
            inp = dropout(ques_w_rep, keep_prob=config.keep_prob, is_train=self.is_train)
            ques_low_h, _ = birnn(cell_fw=f_read_q_low, cell_bw=b_read_q_low,
                                  inputs=inp, dtype=tf.float32,
                                  scope='ques_low_under',
                                  sequence_length=self.q_len)
            ques_low_h = tf.concat(ques_low_h, axis=2)

            # Hqh
            f_read_q_high = tf.contrib.rnn.LSTMCell(config.reading_rep_dim // 2)
            b_read_q_high = tf.contrib.rnn.LSTMCell(config.reading_rep_dim // 2)
            inp = dropout(ques_low_h, keep_prob=config.keep_prob, is_train=self.is_train)
            ques_high_h, _ = birnn(cell_fw=f_read_q_high,
                                   cell_bw=b_read_q_high,
                                   inputs=inp,
                                   dtype=tf.float32,
                                   scope='ques_high_under',
                                   sequence_length=self.q_len)
            ques_high_h = tf.concat(ques_high_h, axis=2)

            # Hcl
            f_read_p_low = tf.contrib.rnn.LSTMCell(config.reading_rep_dim // 2)
            b_read_p_low = tf.contrib.rnn.LSTMCell(config.reading_rep_dim // 2)
            inp = dropout(para_enhanced_rep, keep_prob=config.keep_prob, is_train=self.is_train)
            para_low_h, _ = birnn(cell_fw=f_read_p_low,
                                  cell_bw=b_read_p_low,
                                  inputs=inp,
                                  dtype=tf.float32,
                                  scope='para_low_under',
                                  sequence_length=self.c_len)
            para_low_h = tf.concat(para_low_h, axis=2)

            # Hch
            f_read_p_high = tf.contrib.rnn.LSTMCell(config.reading_rep_dim // 2)
            b_read_p_high = tf.contrib.rnn.LSTMCell(config.reading_rep_dim // 2)
            inp = dropout(para_low_h, keep_prob=config.keep_prob, is_train=self.is_train)
            para_high_h, _ = birnn(cell_fw=f_read_p_high,
                                   cell_bw=b_read_p_high,
                                   inputs=inp,
                                   dtype=tf.float32,
                                   scope='para_high_under',
                                   sequence_length=self.c_len)
            para_high_h = tf.concat(para_high_h, axis=2)

        logging.info("Final Question Understanding")

        with tf.variable_scope("final_q_und"):
            f_uq = tf.contrib.rnn.LSTMCell(config.final_ques_under_dim // 2)
            b_uq = tf.contrib.rnn.LSTMCell(config.final_ques_under_dim // 2)
            inp = tf.concat([ques_low_h, ques_high_h], axis=2)
            inp = dropout(inp, keep_prob=config.keep_prob, is_train=self.is_train)
            final_q_und, _ = birnn(cell_fw=f_uq,
                                   cell_bw=b_uq,
                                   inputs=inp,
                                   dtype=tf.float32,
                                   scope='final_q_und',
                                   sequence_length=self.q_len)
            final_q_und = tf.concat(final_q_und, axis=2)

        logging.info("Fusion High level")
        with tf.variable_scope("high_level_fusion"):
            para_HoW = tf.concat([c_emb, ch_emb,
                                  para_low_h, para_high_h],
                                 axis=2)
            ques_HoW = tf.concat([q_emb, qh_emb,
                                  ques_low_h, ques_high_h],
                                 axis=2)

            para_fused_l = fuse_vecotr(para_HoW, ques_HoW,
                                       self.q_mask,
                                       config.sl_att_dim,
                                       keep_prob=config.keep_prob,
                                       is_train=self.is_train,
                                       B=ques_low_h,
                                       scope='low_level_fusion')

            para_fused_h = fuse_vecotr(para_HoW, ques_HoW,
                                       self.q_mask,
                                       config.sh_att_dim,
                                       keep_prob=config.keep_prob,
                                       is_train=self.is_train,
                                       B=ques_high_h,
                                       scope='high_level_fusion')

            para_fused_u = fuse_vecotr(para_HoW, ques_HoW,
                                       self.q_mask,
                                       config.su_att_dim,
                                       keep_prob=config.keep_prob,
                                       is_train=self.is_train,
                                       B=final_q_und,
                                       scope='understanding_fusion')

            inp = tf.concat([para_low_h, para_high_h,
                             para_fused_l, para_fused_h,
                             para_fused_u], axis=2)
            inp = dropout(inp, keep_prob=config.keep_prob, is_train=self.is_train)

            f_vc = tf.contrib.rnn.LSTMCell(config.fully_fused_para_dim // 2)
            b_vc = tf.contrib.rnn.LSTMCell(config.fully_fused_para_dim // 2)
            ff_para, _ = birnn(cell_fw=f_vc, cell_bw=b_vc, inputs=inp,
                               dtype=tf.float32, scope='full_fused_para',
                               sequence_length=self.c_len)
            ff_para = tf.concat(ff_para, axis=2)
        logging.info("Self boosting fusion")

        with tf.variable_scope("self_boosting_fusion"):
            para_HoW = tf.concat([c_emb, ch_emb,
                                  para_low_h, para_high_h,
                                  para_fused_l, para_fused_h,
                                  para_fused_u, ff_para],
                                 axis=2)

            ff_fused_para = fuse_vecotr(para_HoW, para_HoW,
                                        self.q_mask,
                                        config.selfboost_att_dim,
                                        keep_prob=config.keep_prob,
                                        is_train=self.is_train,
                                        B=ff_para,
                                        scope='self_boosted_fusion')
            f_sb = tf.contrib.rnn.LSTMCell(config.selfboost_rep_dim // 2)
            b_sb = tf.contrib.rnn.LSTMCell(config.selfboost_rep_dim // 2)
            inp = tf.concat([ff_para, ff_fused_para], axis=2)
            inp = dropout(inp, keep_prob=config.keep_prob, is_train=self.is_train)
            final_para_rep, _ = birnn(cell_fw=f_sb, cell_bw=b_sb, inputs=inp,
                                      dtype=tf.float32, scope='self_boosted')
            final_para_rep = tf.concat(final_para_rep, axis=2)

        logging.info("Fusion Net construction complete")
        logging.info("SQuAD specific construction begins")
        # now we have U_c, U_q = final_para_rep, final_q_und
        # The rest of the network is for SQuAD
        # TODO: This part is a little confusing
        # 通过embedding q 获得rQ
        with tf.variable_scope("pointer"):
            # rQ
            logging.info("Sumarized question understanding vector")
            init = summ_vector(final_q_und, config.final_ques_under_dim//2, mask=self.q_mask,
                               keep_prob=config.ptr_keep_prob, is_train=self.is_train)
            logging.info("generation the span")
            pointer = ptr_net(batch=N, hidden=init.get_shape().as_list(
            )[-1], keep_prob=config.ptr_keep_prob, is_train=self.is_train)
            logits1, logits2 = pointer(init,
                                       final_para_rep,
                                       init.get_shape().as_list()[-1],
                                       self.c_mask)
        with tf.variable_scope("predict"):
            self.start_logits = tf.nn.softmax(logits1)
            self.stop_logits = tf.nn.softmax(logits2)
            outer = tf.matmul(tf.expand_dims(tf.nn.softmax(logits1), axis=2),
                              tf.expand_dims(tf.nn.softmax(logits2), axis=1))
            outer = tf.matrix_band_part(outer, 0, 15)
            self.yp1 = tf.argmax(tf.reduce_max(outer, axis=2), axis=1)
            self.yp2 = tf.argmax(tf.reduce_max(outer, axis=1), axis=1)
            losses = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits1, labels=self.y1)
            losses2 = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits2, labels=self.y2)
            # change
            self.loss = losses + losses2

    def get_loss(self):
        return self.loss

    def get_global_step(self):
        return self.global_step
