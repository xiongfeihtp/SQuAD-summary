import tensorflow as tf
from func import cudnn_gru, native_gru, dot_attention, summ, dropout, ptr_net
import numpy as np

# Softmax over axis
def softmax(target, axis, mask, epsilon=1e-12, name=None):
    with tf.op_scope([target], name, 'softmax'):
        max_axis = tf.reduce_max(target, axis, keep_dims=True)
        target_exp = tf.exp(target - max_axis) * mask
        normalize = tf.reduce_sum(target_exp, axis, keep_dims=True)
        softmax = target_exp / (normalize + epsilon)
        return softmax
    
#针对GRU的特征初始化参数
def orthogonal_initializer(scale=1.1):
    ''' From Lasagne and Keras. Reference: Saxe et al., http://arxiv.org/abs/1312.6120
    '''
    print('Warning -- You have opted to use the orthogonal_initializer function')

    def _initializer(shape, dtype=tf.float32, partition_info=None):
        flat_shape = (shape[0], np.prod(shape[1:]))
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        # pick the one with the correct shape
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)  # this needs to be corrected to float32
        print('you have initialized one orthogonal matrix.')
        return tf.constant(scale * q[:shape[0], :shape[1]], dtype=tf.float32)
    return _initializer


class Model(object):
    def __init__(self, config, batch, word_mat=None, char_mat=None, trainable=True, opt=True):
        self.config = config
        self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                           initializer=tf.constant_initializer(0), trainable=False)
        # 这个框架非常好理解，学习数据处理的流程
        self.c, self.q, self.ch, self.qh, self.y1, self.y2, self.qa_id = batch.get_next()
        self.is_train = tf.get_variable(
            "is_train", shape=[], dtype=tf.bool, trainable=False)

        #self.word_mat = tf.get_variable("word_mat", initializer=tf.constant(word_mat, dtype=tf.float32), trainable=True)
        # tf.add_to_collection(tf.GraphKeys.TRAINABLE_VARIABLES,self.word_mat)
        #self.char_mat = tf.get_variable("char_mat", initializer=tf.constant(char_mat, dtype=tf.float32), trainable=True)

        self.c_mask = tf.cast(self.c, tf.bool)
        self.q_mask = tf.cast(self.q, tf.bool)
        self.c_len = tf.reduce_sum(tf.cast(self.c_mask, tf.int32), axis=1)
        self.q_len = tf.reduce_sum(tf.cast(self.q_mask, tf.int32), axis=1)
        if opt:
            N, CL = config.batch_size, config.char_limit
            self.c_maxlen = tf.reduce_max(self.c_len)
            self.q_maxlen = tf.reduce_max(self.q_len)
            self.c = tf.slice(self.c, [0, 0], [N, self.c_maxlen])
            self.q = tf.slice(self.q, [0, 0], [N, self.q_maxlen])
            self.c_mask = tf.slice(self.c_mask, [0, 0], [N, self.c_maxlen])
            self.q_mask = tf.slice(self.q_mask, [0, 0], [N, self.q_maxlen])
            #self.ch = tf.slice(self.ch, [0, 0, 0], [N, self.c_maxlen, CL])
            #self.qh = tf.slice(self.qh, [0, 0, 0], [N, self.q_maxlen, CL])
            self.y1 = tf.slice(self.y1, [0, 0], [N, self.c_maxlen])
            self.y2 = tf.slice(self.y2, [0, 0], [N, self.c_maxlen])
        else:
            self.c_maxlen, self.q_maxlen = config.para_limit, config.ques_limit
        #self.ch_len = tf.reshape(tf.reduce_sum(tf.cast(tf.cast(self.ch, tf.bool), tf.int32), axis=2), [-1])
        #self.qh_len = tf.reshape(tf.reduce_sum(tf.cast(tf.cast(self.qh, tf.bool), tf.int32), axis=2), [-1])
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
        N, PL, QL, CL, d, dc, dg = config.batch_size, self.c_maxlen, self.q_maxlen, config.char_limit, config.hidden, config.char_dim, config.char_hidden
        gru = cudnn_gru if config.use_cudnn else native_gru
        with tf.variable_scope("emb"):
            # with tf.variable_scope("char"):
            # ch_emb = tf.reshape(tf.nn.embedding_lookup(
            #     self.char_mat, self.ch), [N * PL, CL, dc])
            # qh_emb = tf.reshape(tf.nn.embedding_lookup(
            #     self.char_mat, self.qh), [N * QL, CL, dc])
            #
            # ch_emb = dropout(
            #     ch_emb, keep_prob=config.keep_prob, is_train=self.is_train)
            # qh_emb = dropout(
            #     qh_emb, keep_prob=config.keep_prob, is_train=self.is_train)
            #
            # cell_fw = tf.contrib.rnn.GRUCell(dg)
            # cell_bw = tf.contrib.rnn.GRUCell(dg)
            #
            # _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
            #     cell_fw, cell_bw, ch_emb, self.ch_len, dtype=tf.float32)
            # ch_emb = tf.concat([state_fw, state_bw], axis=1)
            # _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
            #     cell_fw, cell_bw, qh_emb, self.qh_len, dtype=tf.float32)
            # qh_emb = tf.concat([state_fw, state_bw], axis=1)
            # qh_emb = tf.reshape(qh_emb, [N, QL, 2 * dg])
            # ch_emb = tf.reshape(ch_emb, [N, PL, 2 * dg])
            embedding = tf.get_variable('embedding',
                                        [config.vocab_size, config.embedding_size],
                                        initializer=tf.random_uniform_initializer(minval=-0.05, maxval=0.05))

            self.regularizer = tf.nn.l2_loss(embedding)

            with tf.name_scope("word"):
                c_emb = tf.nn.embedding_lookup(embedding, self.c)
                q_emb = tf.nn.embedding_lookup(embedding, self.q)
            c_emb = dropout(c_emb, keep_prob=config.keep_prob, is_train=self.is_train)
            q_emb = dropout(q_emb, keep_prob=config.keep_prob, is_train=self.is_train)
            c_emb = tf.reshape(c_emb, [N, PL, config.embedding_size])
            q_emb = tf.reshape(q_emb, [N, QL, config.embedding_size])
            #     c_emb = tf.nn.embedding_lookup(self.word_mat, self.c)
            #     q_emb = tf.nn.embedding_lookup(self.word_mat, self.q)
            # c_emb = tf.concat([c_emb, ch_emb], axis=2)
            # q_emb = tf.concat([q_emb, qh_emb], axis=2)

        with tf.variable_scope("encoding"):
            # 1层 lstm对输出进行编码
            rnn_c = gru(num_layers=1, num_units=d, batch_size=N, input_size=c_emb.get_shape(
            ).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)
            rnn_q = gru(num_layers=1, num_units=d, batch_size=N, input_size=q_emb.get_shape(
            ).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)
            c = rnn_c(c_emb, seq_len=self.c_len)
            q = rnn_q(q_emb, seq_len=self.q_len)


        with tf.variable_scope("attention"):
            M = tf.matmul(c, q, adjoint_b=True)
            M_mask = tf.to_float(tf.matmul(tf.cast(tf.expand_dims(self.c_mask, -1),tf.int32), tf.cast(tf.expand_dims(self.q_mask, 1),tf.int32)))
            alpha = softmax(M, 1, M_mask)  # (batch_size,M,N)
            beta = softmax(M, 2, M_mask)   # (batch_size,M,N)
            # query_importance = tf.expand_dims(tf.reduce_mean(beta, reduction_indices=1), -1)
            query_importance = tf.expand_dims(tf.reduce_sum(beta, 1) / tf.to_float(tf.expand_dims(PL, -1)), -1)# (batch_size,N,1)
            s = tf.squeeze(tf.matmul(alpha, query_importance), [2])# (batch_size,M)
            #unpacked_s = zip(tf.unstack(s, config.batch_size), tf.unstack(self.c, config.batch_size))
            #y_hat=(batch_size,config.vocab_size)  (代表每个词为答案的概率)
            #y_hat = tf.stack([tf.unsorted_segment_sum(attentions, sentence_ids, config.vocab_size) for (attentions, sentence_ids) in unpacked_s])
            match=c*tf.reshape(s,[-1,PL,1])   #(batch_size,max_c_len,dim)
        #通过embedding q 获得rQ
        with tf.variable_scope("pointer"):
            init = summ(q[:, :, -2 * d:], d, mask=self.q_mask,
                        keep_prob=config.ptr_keep_prob, is_train=self.is_train)
            pointer = ptr_net(batch=N, hidden=init.get_shape().as_list(
            )[-1], keep_prob=config.ptr_keep_prob, is_train=self.is_train)
            logits1, logits2 = pointer(init, match, d, self.c_mask)


        with tf.variable_scope("predict"):
            self.start_logits=tf.nn.softmax(logits1)
            self.stop_logits=tf.nn.softmax(logits2)
            outer = tf.matmul(tf.expand_dims(tf.nn.softmax(logits1), axis=2),
                              tf.expand_dims(tf.nn.softmax(logits2), axis=1))
            outer = tf.matrix_band_part(outer, 0, 15)
            self.yp1 = tf.argmax(tf.reduce_max(outer, axis=2), axis=1)
            self.yp2 = tf.argmax(tf.reduce_max(outer, axis=1), axis=1)
            losses = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits1, labels=self.y1)
            losses2 = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits2, labels=self.y2)
            self.loss = tf.reduce_mean(losses + losses2)+config.l2_reg * self.regularizer
        #             qc_att = dot_attention(c, q, mask=self.q_mask, hidden=d,
        #                                    keep_prob=config.keep_prob, is_train=self.is_train)
        #             rnn = gru(num_layers=1, num_units=d, batch_size=N, input_size=qc_att.get_shape(
        #             ).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)
        #             att = rnn(qc_att, seq_len=self.c_len)
        #
        #
        #         with tf.variable_scope("match"):
        #             self_att = dot_attention(
        #                 att, att, mask=self.c_mask, hidden=d, keep_prob=config.keep_prob, is_train=self.is_train)
        #             rnn = gru(num_layers=1, num_units=d, batch_size=N, input_size=self_att.get_shape(
        #             ).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)
        #             match = rnn(self_att, seq_len=self.c_len)
        #
        # #通过embedding q 获得rQ
        #         with tf.variable_scope("pointer"):
        #             init = summ(q[:, :, -2 * d:], d, mask=self.q_mask,
        #                         keep_prob=config.ptr_keep_prob, is_train=self.is_train)
        #             pointer = ptr_net(batch=N, hidden=init.get_shape().as_list(
        #             )[-1], keep_prob=config.ptr_keep_prob, is_train=self.is_train)
        #             logits1, logits2 = pointer(init, match, d, self.c_mask)

        # with tf.variable_scope("predict"):
        #     self.start_logits=tf.nn.softmax(logits1)
        #     self.stop_logits=tf.nn.softmax(logits2)
        #     outer = tf.matmul(tf.expand_dims(tf.nn.softmax(logits1), axis=2),
        #                       tf.expand_dims(tf.nn.softmax(logits2), axis=1))
        #     outer = tf.matrix_band_part(outer, 0, 15)
        #     self.yp1 = tf.argmax(tf.reduce_max(outer, axis=2), axis=1)
        #     self.yp2 = tf.argmax(tf.reduce_max(outer, axis=1), axis=1)
        #     losses = tf.nn.softmax_cross_entropy_with_logits(
        #         logits=logits1, labels=self.y1)
        #     losses2 = tf.nn.softmax_cross_entropy_with_logits(
        #         logits=logits2, labels=self.y2)
        #     self.loss = tf.reduce_mean(losses + losses2)
    def get_loss(self):
        return self.loss

    def get_global_step(self):
        return self.global_step
