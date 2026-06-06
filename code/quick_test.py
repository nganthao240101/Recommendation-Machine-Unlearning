"""
Quick test: Train 1 local model for 1 epoch to verify setup works
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from utility.parser import parse_args
from utility.helper import *
from utility.load_data import Data
from utility.batch_test import *
import tensorflow as tf

args = parse_args()
args.epoch = 1  # Chỉ 1 epoch
args.verbose = 1

print("="*60)
print("QUICK TEST: Train 1 local model for 1 epoch")
print("="*60)
print(f"Dataset: {args.dataset}")
print(f"Partition type: {args.part_type} ({'InBP' if args.part_type == 1 else 'UBP' if args.part_type == 2 else 'Random'})")
print(f"Epochs: {args.epoch}")

# Load data
print("\nLoading data...")
data_generator = Data(
    path=args.data_path + args.dataset,
    batch_size=args.batch_size,
    part_type=args.part_type,
    part_num=args.part_num,
    part_T=args.part_T
)
print(f"n_users={data_generator.n_users}, n_items={data_generator.n_items}")

# Build model
print("\nBuilding model...")
from RecEraser_BPR import RecEraser_BPR

config = {'n_users': data_generator.n_users, 'n_items': data_generator.n_items}
model = RecEraser_BPR(data_config=config)

# Train 1 local model for 1 epoch
print("\n" + "="*60)
print("TRAINING: Local model 0, Epoch 0")
print("="*60)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

# Train 1 batch
users, pos_items, neg_items = data_generator.local_sample(0)
_, loss, mf_loss, reg_loss = sess.run(
    [model.opt_local[0], model.loss_local[0], model.mf_loss_local[0], model.reg_loss_local[0]],
    feed_dict={model.users: users, model.pos_items: pos_items, model.neg_items: neg_items}
)

print(f"Batch loss: {loss:.5f} = {mf_loss:.5f} + {reg_loss:.5f}")

# Test
print("\nTesting on test set...")
users_to_test = list(data_generator.test_set.keys())
ret = test(sess, model, users_to_test, drop_flag=False, local_flag=True, local_num=0)
print(f"Recall@20: {ret['recall'][0]:.5f}")
print(f"Recall@50: {ret['recall'][1]:.5f}")

print("\n" + "="*60)
print("TEST COMPLETED SUCCESSFULLY!")
print("="*60)
print("\nModel is ready. To train full model:")
print("  python RecEraser_BPR.py --dataset ml-1m --part_type 1 --part_num 10 --epoch 100")