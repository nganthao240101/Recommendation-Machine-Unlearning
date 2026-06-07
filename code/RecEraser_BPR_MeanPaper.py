"""
Paper-faithful Mean aggregation for RecEraser (no trans_W, no attention).

This is the simplest aggregation baseline described in Section 4.4 of
Chen et al., WWW'22. For each (user, item):
  p_u = (1 / n_local) * sum_k p_u^(k)        (mean of local user embeddings)
  q_i = (1 / n_local) * sum_k q_i^(k)        (mean of local item embeddings)
  y_hat = p_u^T q_i

The local models are trained exactly as in RecEraser_BPR.py (so the
checkpoint format is identical and partition files C, C_U, C_I can be
re-used). Only the aggregator changes.

Usage:
  python RecEraser_BPR_MeanPaper.py --dataset ml-1m --part_type 1 \
                                    --part_num 5 --epoch 30
  python RecEraser_BPR_MeanPaper.py --dataset ml-1m --part_type 2 \
                                    --part_num 5 --epoch 30
  ...
"""
import tensorflow as tf
from utility.helper import *
import numpy as np
from scipy.sparse import csr_matrix
from utility.batch_test import *
import os
import sys
import pickle
import copy
import random


class RecEraser_BPR_MeanPaper(object):
    def __init__(self, data_config):
        self.model_type = 'RecEraser_BPR_MeanPaper'
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.lr = args.lr

        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size
        self.weight_size = eval(args.layer_size)
        self.n_layers = len(self.weight_size)
        self.regs = eval(args.regs)
        self.decay = self.regs[0]
        self.verbose = args.verbose
        self.Ks = eval(args.Ks)
        self.num_local = args.part_num

        self.weights = self._init_weights()

        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.placeholder(tf.int32, shape=(None,))
        self.neg_items = tf.placeholder(tf.int32, shape=(None,))
        self.dropout_keep_prob = tf.placeholder(tf.float32,
                                                name='dropout_keep_prob")

        # Per-local BPR losses (same as RecEraser_BPR)
        self.opt_local = []
        self.loss_local = []
        self.mf_loss_local = []
        self.reg_loss_local = []
        self.batch_ratings_local = []
        for i in range(self.num_local):
            line = self.train_single_model(i)
            self.opt_local.append(line[0])
            self.loss_local.append(line[1])
            self.mf_loss_local.append(line[2])
            self.reg_loss_local.append(line[3])
            self.batch_ratings_local.append(line[4])

        # Mean aggregator
        line = self.train_agg_model_mean()
        self.opt_agg = line[0]
        self.loss_agg = line[1]
        self.mf_loss_agg = line[2]
        self.reg_loss_agg = line[3]
        self.attention_loss = line[4]
        self.batch_ratings = line[5]
        self.u_w = line[6]

    def _init_weights(self):
        all_weights = dict()
        initializer = tf.contrib.layers.xavier_initializer()

        all_weights['user_embedding'] = tf.Variable(
            initializer([self.n_users, self.num_local, self.emb_dim]),
            name='user_embedding')
        all_weights['item_embedding'] = tf.Variable(
            initializer([self.n_items, self.num_local, self.emb_dim]),
            name='item_embedding')
        return all_weights

    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)
        regularizer = (tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items)
                       + tf.nn.l2_loss(neg_items)) / self.batch_size
        diff = tf.clip_by_value(pos_scores - neg_scores, -50.0, 50.0)
        mf_loss = tf.reduce_mean(tf.nn.softplus(-diff))
        reg_loss = self.decay * regularizer
        return mf_loss, reg_loss

    def train_single_model(self, local_num):
        u_e = tf.nn.embedding_lookup(
            self.weights['user_embedding'][:, local_num], self.users)
        pos_i_e = tf.nn.embedding_lookup(
            self.weights['item_embedding'][:, local_num], self.pos_items)
        neg_i_e = tf.nn.embedding_lookup(
            self.weights['item_embedding'][:, local_num], self.neg_items)
        mf_loss, reg_loss = self.create_bpr_loss(u_e, pos_i_e, neg_i_e)
        loss = mf_loss + reg_loss
        batch_ratings = tf.matmul(u_e, pos_i_e, transpose_a=False,
                                  transpose_b=True)
        opt = tf.train.AdagradOptimizer(
            learning_rate=self.lr, initial_accumulator_value=1e-8
        ).minimize(loss)
        return opt, loss, mf_loss, reg_loss, batch_ratings

    def train_agg_model_mean(self):
        """Paper-faithful mean aggregation (no trans_W, no attention)."""
        u_es = tf.nn.embedding_lookup(self.weights['user_embedding'],
                                      self.users)
        pos_i_es = tf.nn.embedding_lookup(self.weights['item_embedding'],
                                          self.pos_items)
        neg_i_es = tf.nn.embedding_lookup(self.weights['item_embedding'],
                                          self.neg_items)

        # Simple mean across the num_local axis (index 1)
        u_e = tf.reduce_mean(u_es, axis=1)
        pos_i_e = tf.reduce_mean(pos_i_es, axis=1)
        neg_i_e = tf.reduce_mean(neg_i_es, axis=1)

        u_e_drop = tf.nn.dropout(u_e, self.dropout_keep_prob)
        mf_loss, reg_loss = self.create_bpr_loss(u_e_drop, pos_i_e, neg_i_e)
        batch_ratings = tf.matmul(u_e, pos_i_e, transpose_a=False,
                                  transpose_b=True)
        loss = mf_loss + reg_loss
        opt = tf.train.AdagradOptimizer(
            learning_rate=self.lr, initial_accumulator_value=1e-8
        ).minimize(loss)

        # Return dummy attention loss / u_w for interface compat
        return (opt, loss, mf_loss, reg_loss,
                tf.constant(0.0), batch_ratings, tf.zeros([1, 1, 1]))


if __name__ == '__main__':
    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items

    t0 = time()
    model = RecEraser_BPR_MeanPaper(data_config=config)
    saver = tf.train.Saver()

    weights_save_path = (
        '%sweights/%s/%s/num-%s_type-%s_r%s' %
        (args.proj_path, args.dataset, model.model_type,
         str(args.part_num), str(args.part_type),
         '-'.join([str(r) for r in eval(args.regs)]))
    )
    ensureDir(weights_save_path)
    save_saver = tf.train.Saver(max_to_keep=1)

    config_sess = tf.ConfigProto()
    config_sess.gpu_options.allow_growth = True
    sess = tf.Session(config=config_sess)
    sess.run(tf.global_variables_initializer())

    # Train local models (same protocol as RecEraser_BPR.py)
    for i in range(args.part_num):
        cur_best_pre_0 = 0.
        stopping_step = 0
        for epoch in range(args.epoch):
            t1 = time()
            loss, mf_loss, reg_loss = 0., 0., 0.
            n_batch = data_generator.n_C[i] // args.batch_size + 1
            for idx in range(n_batch):
                users, pos_items, neg_items = data_generator.local_sample(i)
                _, batch_loss, batch_mf_loss, batch_reg_loss = sess.run(
                    [model.opt_local[i], model.loss_local[i],
                     model.mf_loss_local[i], model.reg_loss_local[i]],
                    feed_dict={model.users: users,
                               model.pos_items: pos_items,
                               model.neg_items: neg_items})
                loss += batch_loss
                mf_loss += batch_mf_loss
                reg_loss += batch_reg_loss
            if np.isnan(loss):
                print('ERROR: loss is nan.'); sys.exit()
            if (epoch + 1) % 5 != 0:
                continue
            t2 = time()
            users_to_test = list(data_generator.test_set.keys())
            ret = test(sess, model, users_to_test, drop_flag=False,
                       local_flag=True, local_num=i)
            t3 = time()
            if args.verbose > 0:
                perf_str = (
                    '[local %d] Epoch %d [%.1fs+%.1fs]: train==[%.5f=%.5f+%.5f],'
                    ' recall=[%.5f,%.5f], precision=[%.5f,%.5f],'
                    ' ndcg=[%.5f,%.5f]' %
                    (i, epoch, t2 - t1, t3 - t2, loss, mf_loss, reg_loss,
                     ret['recall'][0], ret['recall'][1],
                     ret['precision'][0], ret['precision'][1],
                     ret['ndcg'][0], ret['ndcg'][1]))
                print(perf_str)
            cur_best_pre_0, stopping_step, should_stop = early_stopping(
                ret['recall'][0], cur_best_pre_0, stopping_step,
                expected_order='acc', flag_step=10)
            if should_stop:
                break

    save_saver.save(sess, weights_save_path + '/weights')
    print('save the weights in path: ', weights_save_path)

    # Train aggregator (mean over local embeddings)
    cur_best_pre_0 = 0.
    stopping_step = 0
    for epoch in range(args.epoch_agg):
        t1 = time()
        loss, mf_loss, reg_loss, attn = 0., 0., 0., 0.
        n_batch = data_generator.n_train // args.batch_size + 1
        for idx in range(n_batch):
            users, pos_items, neg_items = data_generator.sample()
            _, batch_loss, batch_mf_loss, batch_reg_loss = sess.run(
                [model.opt_agg, model.loss_agg,
                 model.mf_loss_agg, model.reg_loss_agg],
                feed_dict={model.users: users,
                           model.pos_items: pos_items,
                           model.neg_items: neg_items,
                           model.dropout_keep_prob: args.dropout})
            loss += batch_loss
            mf_loss += batch_mf_loss
            reg_loss += batch_reg_loss
        if np.isnan(loss):
            print('ERROR: loss is nan.'); sys.exit()
        if (epoch + 1) % 1 != 0:
            continue
        t2 = time()
        users_to_test = list(data_generator.test_set.keys())
        ret = test(sess, model, users_to_test, drop_flag=False)
        t3 = time()
        if args.verbose > 0:
            print(
                'Epoch %d [%.1fs+%.1fs]: train==[%.5f=%.5f+%.5f],'
                ' recall=[%.5f,%.5f,%.5f], precision=[%.5f,%.5f,%.5f],'
                ' ndcg=[%.5f,%.5f,%.5f]' %
                (epoch, t2 - t1, t3 - t2, loss, mf_loss, reg_loss,
                 ret['recall'][0], ret['recall'][1], ret['recall'][2],
                 ret['precision'][0], ret['precision'][1],
                 ret['precision'][2],
                 ret['ndcg'][0], ret['ndcg'][1], ret['ndcg'][2]))
        cur_best_pre_0, stopping_step, should_stop = early_stopping(
            ret['recall'][0], cur_best_pre_0, stopping_step,
            expected_order='acc', flag_step=10)
        if should_stop:
            break
