"""
Early stopping on a validation split, as the authors' training loops do it.

The authors' RecEraser_BPR.py trains up to --epoch 1000, evaluates every 5
epochs and stops after 10 evaluations without a better Recall@20
(flag_step=10); the aggregator runs up to --epoch_agg 50, evaluated every
epoch with the same rule. They evaluate on the TEST set. Here the rule is
the same but it watches a VALIDATION split held out of the training data,
so the test set is touched only once, for the reported numbers.

  split_validation(train, ratio, seed) -> (train_fit, val)
  configure(...)                        -> turns it on for the run
  EarlyStopper(...)                     -> the epoch_end callback of a loop

A training loop calls the stopper after every epoch; it returns True when
the loop should stop. finish() puts the best weights back. LAST records the
epochs each model of the last run settled on, so that unlearning can retrain
an affected shard for that many epochs (no evaluation inside the timed
retrain) and the work column can count them.
"""

import random

import numpy as np

CFG = {
    'on': False,
    'every': 5,           # evaluate every N epochs (authors: 5)
    'patience': 10,       # evaluations without improvement (authors: flag_step=10)
    'max_epochs': 1000,   # authors: --epoch 1000
    'agg_every': 1,       # aggregator: every epoch (authors)
    'agg_patience': 10,
    'agg_max_epochs': 50,  # authors: --epoch_agg 50
    'k': 20,              # Recall@20 (authors: ret['recall'][0], Ks=[20, ...])
    'val': None,          # {user: [val items]}
    'unlearn': False,     # the retrain of an unlearning request stops early too
}

# epochs chosen in the last training of a run: {'shards': {sid: ep},
# 'global': ep, 'agg': ep}
LAST = {}


def on():
    return CFG['on']


def configure(val, every=5, patience=10, max_epochs=1000, agg_patience=10,
              agg_max_epochs=50, k=20, unlearn=False):
    CFG.update(on=True, val=val, every=every, patience=patience,
               max_epochs=max_epochs, agg_patience=agg_patience,
               agg_max_epochs=agg_max_epochs, k=k, unlearn=unlearn)


def unlearn_on():
    """Retrain after a deletion with the same early stopping as the
    training (else: for the epochs the trained model settled on)."""
    return CFG['on'] and CFG['unlearn']


def describe():
    d = describe_train()
    if not CFG['on']:
        return d
    return d + "; unlearning retrain: " + ("early stopping too, best weights" if CFG['unlearn']
                                          else "the epochs the trained model settled on")


def describe_train():
    """The training part only (it decides a saved model, the unlearning
    retrain does not)."""
    if not CFG['on']:
        return 'off (fixed epochs)'
    n = sum(len(v) for v in CFG['val'].values())
    return (f"on: validation {n} interactions of {len(CFG['val'])} users, "
            f"Recall@{CFG['k']} every {CFG['every']} epochs, patience "
            f"{CFG['patience']} evaluations, max {CFG['max_epochs']} epochs; "
            f"aggregator every {CFG['agg_every']}, patience {CFG['agg_patience']}, "
            f"max {CFG['agg_max_epochs']}")


def split_validation(train, ratio=0.1, seed=2024):
    """Hold out `ratio` of each user's training interactions (at least one,
    and always leaving one to train on; users with a single interaction keep
    it). Fixed seed: the same split for every arm and every run seed."""
    rng = random.Random(seed)
    fit, val = {}, {}
    for u in sorted(train):
        items = list(train[u])
        if len(items) < 2:
            fit[u] = items
            continue
        n_val = min(len(items) - 1, max(1, int(round(ratio * len(items)))))
        idx = set(rng.sample(range(len(items)), n_val))
        val[u] = [it for i, it in enumerate(items) if i in idx]
        fit[u] = [it for i, it in enumerate(items) if i not in idx]
    return fit, val


class EarlyStopper:
    """epoch_end callback: evaluate on the validation users, keep the best
    weights, say when to stop.

    score_fn(user_tensor) -> scores over all items; train_data gives the items
    to exclude from the ranking (the model's own training interactions);
    users are the validation users this model answers for. state_fn() returns
    a copy of the weights, load_fn(state) puts it back.
    """

    def __init__(self, score_fn, users, train_data, n_items, device, state_fn,
                 load_fn, every=None, patience=None, name=''):
        self.score_fn, self.train_data = score_fn, train_data
        self.n_items, self.device = n_items, device
        self.state_fn, self.load_fn = state_fn, load_fn
        val = CFG['val'] or {}
        self.val = {u: val[u] for u in users if val.get(u)}
        self.users = sorted(self.val)
        self.every = every or CFG['every']
        self.patience = patience or CFG['patience']
        self.name = name
        self.best, self.best_epoch, self.bad = -1.0, 0, 0
        self.best_state, self.history, self.last_epoch = None, [], 0

    def __call__(self, ep):
        self.last_epoch = ep + 1
        if (ep + 1) % self.every:
            return False
        if not self.users:                 # nothing to validate on: no stopping
            self.best_epoch = ep + 1
            return False
        import fast_train
        r = fast_train.evaluate_topk(self.score_fn, self.users, self.train_data,
                                     self.val, self.n_items, self.device,
                                     Ks=(CFG['k'],))['recall'][0]
        self.history.append((ep + 1, r))
        if r > self.best:
            self.best, self.best_epoch, self.bad = r, ep + 1, 0
            self.best_state = self.state_fn()
        else:
            self.bad += 1
        return self.bad >= self.patience

    def finish(self):
        """Back to the best weights; -> the epoch they come from."""
        if self.best_state is not None:
            self.load_fn(self.best_state)
        if not self.best_epoch:
            self.best_epoch = self.last_epoch
        return self.best_epoch


def with_val(data, deleted=()):
    """data plus each user's validation items: what a test ranking must
    exclude once the validation items were held out of training."""
    val = CFG['val'] or {}
    out = {}
    for u, it in data.items():
        if u in deleted:
            continue
        out[u] = list(it) + list(val.get(u, ()))
    return out


def model_stopper(model, users, train_data, n_items, device, name=''):
    """EarlyStopper for a model with user_embedding / item_embedding tables
    (BPRMF, WMF)."""
    def score(ut):
        return model.user_embedding.weight[ut] @ model.item_embedding.weight.T

    def state():
        return {k: v.detach().clone() for k, v in model.state_dict().items()}
    return EarlyStopper(score, users, train_data, n_items, device, state,
                        model.load_state_dict, name=name)


def simulate(history, patience):
    """Replay a validation curve [(epoch, recall)] under a patience: ->
    (epoch it would stop at, best epoch by then, best recall by then)."""
    best, best_ep, bad = -1.0, 0, 0
    for ep, r in history:
        if r > best:
            best, best_ep, bad = r, ep, 0
        else:
            bad += 1
        if bad >= patience:
            return ep, best_ep, best
    ep = history[-1][0] if history else 0
    return ep, best_ep, best


def mean(xs):
    return float(np.mean(xs)) if xs else 0.0
