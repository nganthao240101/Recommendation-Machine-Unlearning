"""
SISA as defined by Bourtoule et al. (Sharded, Isolated, Sliced, Aggregated):

  1. Sharding    the users are randomly permuted (run seed) and split evenly
                 into K disjoint shards;
  2. Isolation   one model per shard, no parameter or data shared;
  3. Slicing     each shard's users are split into R slices; the model is
                 trained incrementally - stage r trains on slices 1..r,
                 starting from the checkpoint of stage r-1 - and a checkpoint
                 (weights and optimiser state) is kept after every stage.
                 Deleting a user of slice j restores checkpoint j-1 (a fresh
                 model when j = 1) and retrains stages j..R on the retained
                 users only;
  4. Aggregation the K models' scores are averaged (the ranking analogue of
                 the majority vote).

Epochs per stage: with early stopping every stage trains with it on the
validation items of the users it holds and keeps its best weights;
without, every stage trains 2E/(R+1) epochs (the SISA paper's choice, so the
R stages together cost about one training of E epochs on the whole shard).
"""

import random
import time

import numpy as np
import torch

import early_stop
import fast_train
import wmf_backbone


def _clone(sd):
    return {k: (v.detach().to('cpu').clone() if torch.is_tensor(v) else v)
            for k, v in sd.items()}


def _opt_clone(opt):
    st = opt.state_dict()
    return {'state': {k: _clone(v) for k, v in st['state'].items()},
            'param_groups': [dict(g) for g in st['param_groups']]}


def _train(model, opt, data, n_items, device, batch_size, n_epochs, epoch_end=None):
    """Train `model` with the given optimiser (kept across stages). -> samples/epoch."""
    if not any(data.values()):
        return 0
    if isinstance(model, wmf_backbone.WMF):
        _, n = wmf_backbone.train_wmf(lambda u: model.user_embedding(u),
                                      lambda: model.item_embedding.weight,
                                      list(model.parameters()), data, device, None,
                                      n_epochs, epoch_end=epoch_end, optimizer=opt)
        return n
    if fast_train.use_gpu_engine():
        samples = fast_train.draw_samples(data, n_items)

        def step(u, p, n):
            opt.zero_grad()
            loss = model(u, p, n)
            loss.backward()
            opt.step()
            return loss
        fast_train.train_epochs(step, samples, device, batch_size, n_epochs,
                                epoch_end=epoch_end)
        fast_train.sync(device)
        return len(samples[0])
    samples = []
    for u, items in data.items():
        for pos in items:
            neg = random.randint(0, n_items - 1)
            while neg in items:
                neg = random.randint(0, n_items - 1)
            samples.append((u, pos, neg))
    for ep in range(n_epochs):
        random.shuffle(samples)
        nb = max(1, len(samples) // batch_size)
        for b in range(nb):
            chunk = samples[b * batch_size:(b + 1) * batch_size]
            if not chunk:
                continue
            us = torch.LongTensor([c[0] for c in chunk]).to(device)
            ps = torch.LongTensor([c[1] for c in chunk]).to(device)
            ns = torch.LongTensor([c[2] for c in chunk]).to(device)
            loss = model(us, ps, ns)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch_end is not None and epoch_end(ep):
            break
    return len(samples)


def slices_of(shard_users, n_slices, seed, sid):
    """Random split of a shard's users into R slices (fixed by seed and shard)."""
    users = sorted(shard_users)
    random.Random(seed * 1009 + sid).shuffle(users)
    return [users[r::n_slices] for r in range(n_slices)]


def _stage(model, opt, data, n_items, device, batch_size, lr, epochs, n_slices,
           es_on, fixed_epochs=None):
    """One incremental stage. -> (epochs run, best epoch, samples/epoch)."""
    if es_on and fixed_epochs is None:
        es = early_stop.model_stopper(model, list(data), data, n_items, device)
        n = _train(model, opt, data, n_items, device, batch_size,
                   early_stop.CFG['max_epochs'], epoch_end=es)
        best = es.finish()
        return es.last_epoch, best, n, es.eval_time
    ep = fixed_epochs if fixed_epochs is not None else max(1, round(2 * epochs / (n_slices + 1)))
    n = _train(model, opt, data, n_items, device, batch_size, ep)
    return ep, ep, n, 0.0


def run_sisa_arm(train_items, test_data, n_users, n_items, n_shards, emb_dim,
                 user_to_shard, del_users, epochs, lr, batch_size, device, seed,
                 n_slices=5, cache=None, eval_fn=None):
    """Same tuple as run_sharded_arm. eval_fn(shard_models, excl, test) -> res."""
    from exp_partition_quality import set_all_seeds, build_shard_data
    from method_4_ours import ShardModels
    set_all_seeds(seed)
    es_on = early_stop.on()
    shard_data = build_shard_data(train_items, user_to_shard, n_shards)
    slices = {s: slices_of(shard_data[s], n_slices, seed, s) for s in range(n_shards)}
    shard_models = ShardModels(n_users, n_items, emb_dim, n_shards)

    def new_opt(model):
        return torch.optim.Adagrad(model.parameters(), lr=lr, initial_accumulator_value=1e-8)

    def cum(sid, r, data):
        keep = set(u for sl in slices[sid][:r + 1] for u in sl)
        return {u: data[u] for u in keep if u in data}

    if cache and 'sisa' in cache:
        ck = cache['sisa']
        for sid in range(n_shards):
            m = shard_models.create_model(sid, device)
            m.load_state_dict(ck[sid]['stages'][-1]['model'])
        train_time = cache['train_time']
        set_all_seeds(cache['rng_seed'])
        print(f"       model: SISA shards and their {n_slices} slice checkpoints reused "
              f"(trained once), train time {train_time:.1f}s")
    else:
        ck = {}
        t0 = time.time()
        for sid in range(n_shards):
            m = shard_models.create_model(sid, device)
            opt = new_opt(m)
            stages = []
            for r in range(n_slices):
                ran, best, n, _ = _stage(m, opt, cum(sid, r, shard_data[sid]), n_items,
                                         device, batch_size, lr, epochs, n_slices, es_on)
                stages.append({'model': _clone(m.state_dict()), 'opt': _opt_clone(opt),
                               'ran': ran, 'best': best})
            ck[sid] = {'stages': stages}
        train_time = time.time() - t0
        print(f"       SISA: {n_slices} slices per shard, stage epochs (best) "
              + "; ".join(f"s{s}=" + "/".join(str(st['best']) for st in ck[s]['stages'])
                          for s in sorted(ck)))
        if cache is not None:
            cache['sisa'], cache['train_time'] = ck, train_time

    # ---- unlearn: from the first slice that held a deleted user
    retained = {u: it for u, it in train_items.items() if u not in del_users}
    retained_shard = build_shard_data(retained, user_to_shard, n_shards)
    affected = sorted({int(user_to_shard[u]) for u in del_users if u < len(user_to_shard)})
    per_shard_time, per_shard_samples, first_slice = {}, {}, {}
    work, eval_time = 0, 0.0
    t0 = time.time()
    for sid in affected:
        j = min(r for r, sl in enumerate(slices[sid]) for u in sl if u in del_users)
        first_slice[sid] = j + 1
        m = shard_models.create_model(sid, device)
        opt = new_opt(m)
        if j > 0:
            m.load_state_dict(ck[sid]['stages'][j - 1]['model'])
            opt.load_state_dict(ck[sid]['stages'][j - 1]['opt'])
        ts = time.time()
        samples = 0
        for r in range(j, n_slices):
            fixed = None if early_stop.unlearn_on() or not es_on else ck[sid]['stages'][r]['best']
            ran, best, n, ev = _stage(m, opt, cum(sid, r, retained_shard[sid]), n_items,
                                      device, batch_size, lr, epochs, n_slices, es_on, fixed)
            samples += n * ran
            eval_time += ev
        per_shard_time[sid] = time.time() - ts
        per_shard_samples[sid] = samples
        work += samples
    unlearn_time = time.time() - t0
    parallel_time = max(per_shard_time.values()) if per_shard_time else 0.0
    # the work of a sliced retrain is counted directly (samples x epochs of every stage)
    early_stop.LAST = {'shards': {s: 1 for s in affected}, 'work_samples': work}
    if es_on and early_stop.unlearn_on():
        early_stop.LAST['eval_time'] = eval_time
    print(f"       SISA unlearn: retrained from slice "
          + ", ".join(f"s{s}:{first_slice[s]}/{n_slices}" for s in affected))

    excl = early_stop.with_val(retained) if es_on else retained
    test_retained = {u: it for u, it in test_data.items() if u not in del_users}
    res = eval_fn(shard_models, excl, test_retained)
    return (res, train_time, unlearn_time, affected, shard_models, parallel_time,
            per_shard_time, per_shard_samples)
