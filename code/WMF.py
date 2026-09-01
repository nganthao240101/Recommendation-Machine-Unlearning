import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from utility.helper import *
import numpy as np
from scipy.sparse import csr_matrix
from utility.batch_test import *
import os
import sys
import copy
import pickle

class WMF:
    def __init__(self, user_num, item_num, max_item_pu):
        self.user_num = user_num
        self.item_num = item_num
        self.embedding_size = args.embed_size
        self.max_item_pu = max_item_pu
        self.weight1 = args.negative_weight
        self.lambda_bilinear = [0, 0]
        self.lr = args.lr
        self.Ks = eval(args.Ks)

    def _create_placeholders(self):
        self.input_u = tf.compat.v1.placeholder(tf.int32, [None, 1], name="input_uid")
        self.input_ur = tf.compat.v1.placeholder(tf.int32, [None, None], name="input_ur")
        self.dropout_keep_prob = tf.compat.v1.placeholder(tf.float32, name="dropout_keep_prob")

        self.users = tf.compat.v1.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.compat.v1.placeholder(tf.int32, shape=(None,))

    def _create_variables(self):
        self.uidW = tf.Variable(tf.random.truncated_normal(shape=[self.user_num, self.embedding_size], mean=0.0,
                                                    stddev=0.01), dtype=tf.float32, name="uidWg")
        self.iidW = tf.Variable(tf.random.truncated_normal(shape=[self.item_num + 1, self.embedding_size], mean=0.0,
                                                    stddev=0.01), dtype=tf.float32, name="iidW")

    def _create_inference(self):
        self.uid = tf.nn.embedding_lookup(self.uidW, self.input_u)
        self.uid = tf.reshape(self.uid, [-1, self.embedding_size])

        self.uid = tf.nn.dropout(self.uid, self.dropout_keep_prob)

        self.pos_item = tf.nn.embedding_lookup(self.iidW, self.input_ur)
        self.pos_num_r = tf.cast(tf.not_equal(self.input_ur, self.item_num), 'float32')
        self.pos_item = tf.einsum('ab,abc->abc', self.pos_num_r, self.pos_item)
        self.pos_r = tf.einsum('ac,abc->ab', self.uid, self.pos_item)

    def _pre(self):
        u_e = tf.nn.embedding_lookup(self.uidW, self.users)
        pos_i_e = tf.nn.embedding_lookup(self.iidW, self.pos_items)
        self.batch_ratings = tf.matmul(u_e, pos_i_e, transpose_a=False, transpose_b=True)
    def _create_loss(self):
        self.loss1 = self.weight1 * tf.reduce_sum(tf.einsum('ab,ac->bc', self.iidW, self.iidW)
                          * tf.einsum('ab,ac->bc', self.uid, self.uid))
        self.loss1 += tf.reduce_sum((1.0 - self.weight1) * tf.square(self.pos_r) - 2.0 * self.pos_r)
        self.l2_loss0 = tf.nn.l2_loss(self.uidW)
        self.l2_loss1 = tf.nn.l2_loss(self.iidW)
        self.loss = self.loss1 \
                    + self.lambda_bilinear[0] * self.l2_loss0 \
                    + self.lambda_bilinear[1] * self.l2_loss1

        self.reg_loss = self.lambda_bilinear[0] * self.l2_loss0 \
                        + self.lambda_bilinear[1] * self.l2_loss1

        self.opt = tf.compat.v1.train.AdagradOptimizer(learning_rate=self.lr, initial_accumulator_value=1e-8).minimize(self.loss)

    def _build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_inference()
        self._create_loss()
        self._pre()
        #self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)



def get_lables(temp_set, k=0.9999):
    max_item = 0
    item_lenth = []
    for key in temp_set:
        item_lenth.append(len(temp_set[key]))
        if len(temp_set[key]) > max_item:
            max_item = len(temp_set[key])
    print('max_item length of train:', max_item)

    # Find a valid item to pad (use last item in dataset)
    all_items = []
    for key in temp_set:
        all_items.extend(temp_set[key])
    pad_item = max(all_items) if all_items else 0

    result = []
    for key in temp_set:
        temp = temp_set[key]
        tem_length = len(temp)
        labeled = temp.copy()
        if tem_length < max_item:
            labeled.extend([pad_item] * (max_item - tem_length))
        result.append(labeled)

    item_set = []
    for key in temp_set:
        item_set.extend(temp_set[key])
    item_set = set(item_set)

    return max_item, result

def get_sparse_graph(train_items, n_items, item_set):
    row, col, data = [], [], []
    for u, items in train_items.items():
        for i in items:
            row.append(u)
            col.append(n_items + i)
            data.append(1)
    for u, items in train_items.items():
        for i in items:
            row.append(n_items + i)
            col.append(u)
            data.append(1)
    return csr_matrix((data, (row, col)), shape=(len(item_set), len(item_set)))


def prepare_data(train_items):
    n_users = 0
    n_items = 0
    for key in train_items:
        n_users = max(n_users, key)
        for i in train_items[key]:
            n_items = max(n_items, i)
    n_users += 1
    n_items += 1
    train_items_user = {}
    for key in train_items:
        train_items_user[key] = list(train_items[key])
    return n_users, n_items, train_items_user


if __name__ == '__main__':

    tf.random.set_seed(2021)
    np.random.seed(2021)

    n_users, n_items, train_items = prepare_data(data_generator.train_items)
    train = data_generator.train_items
    max_item, lable = get_lables(train_items)

    t0 = time()

    model = WMF(n_users, n_items, max_item)
    model._build_graph()

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    sess.run(tf.compat.v1.global_variables_initializer())
    cur_best_pre_0 = 0.

    run_time = 1


    loss_loger, pre_loger, rec_loger, ndcg_loger,  = [], [], [], []

    for epoch in range(args.epoch):
        loss, pre, rec, ndcg = [], [], [], []
        train_samp = []
        for key in train_items:
            train_samp.append(key)
        random.shuffle(train_samp)
        for user in train_samp:
            user_batch = [user]
            item_batch = list(train_items[user])

            users, items = [], []
            for i in range(args.batch_size):
                users.append(user_batch[i % len(user_batch)])
                items.append(item_batch[i % len(item_batch)])

            _, loss_batch = sess.run(
                [model.opt, model.loss],
                feed_dict={model.input_u: np.array(users).reshape(len(users), 1),
                           model.input_ur: np.array(items).reshape(len(users), max_item),
                           model.dropout_keep_prob: 1,
                           model.users: np.array(users),
                           model.pos_items: np.array(items)})
            loss.append(loss_batch)

        loss_test, recall, ndcg_test = [], [], []
        for user in range(n_users):
            users, items = [], []
            item_batch = list(set(range(n_items)) - set(train_items[user]))

            for i in range(args.batch_size):
                users.append(user)
                items.append(item_batch[i % len(item_batch)])

            ret = sess.run(
                model.batch_ratings,
                feed_dict={model.users: np.array(users),
                           model.pos_items: np.array(items)})
            ret = np.array(ret).flatten()

            item_pos = train_items[user]
            item_score_map = dict(zip(range(len(ret)), ret))
            rank_list = heapq.nlargest(args.Ks[1], item_score_map, key=item_score_map.get)
            item_pop = set(item_pos)

            pre = [0.0 for _ in args.Ks]
            rec = [0.0 for _ in args.Ks]
            ndcg = [0.0 for _ in args.Ks]

            for k in range(len(args.Ks)):
                hit_list = list(rank_list[:args.Ks[k]])
                dcg = 0.0
                for i in range(len(hit_list)):
                    if hit_list[i] in item_pop:
                        dcg += 1.0 / np.log2(i + 2.0)
                idcg = 0.0
                for i in range(min(len(item_pos), args.Ks[k])):
                    idcg += 1.0 / np.log2(i + 2.0)
                ndcg[k] = dcg / idcg

            for k in range(len(args.Ks)):
                hit_list = list(rank_list[:args.Ks[k]])
                hit_num = len(set(hit_list) & set(item_pos))
                pre[k] = hit_num / float(args.Ks[k])
                rec[k] = hit_num / float(len(item_pos))

            pre_loger.append(pre)
            rec_loger.append(rec)
            ndcg_loger.append(ndcg)

        loss_loger.append(np.mean(loss))
        print(f'Epoch {epoch}: loss={np.mean(loss):.4f}')

    uidW, iidW = sess.run([model.uidW, model.iidW])

    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save with timestamp
    with open(args.data_path + args.dataset + f'/user_pretrain_{timestamp}.pk', 'wb') as f:
        pickle.dump(uidW, f)
    with open(args.data_path + args.dataset + f'/item_pretrain_{timestamp}.pk', 'wb') as f:
        pickle.dump(iidW, f)

    # Also save as default
    with open(args.data_path + args.dataset + '/user_pretrain.pk', 'wb') as f:
        pickle.dump(uidW, f)
    with open(args.data_path + args.dataset + '/item_pretrain.pk', 'wb') as f:
        pickle.dump(iidW, f)

    print(f'\n[SAVED] Embeddings with timestamp: {timestamp}')
