from utility.parser import parse_args
import pickle
import copy
import random
import numpy as np
args = parse_args()


def E_score1(a,b):
    return np.sum(a * b) / (np.sqrt(
                    np.sum(np.power(a, 2))) * np.sqrt(np.sum(np.power(b, 2))) )

def E_score2(a,b):
    return np.sum(np.power(a-b, 2))


# Interaction-based Balanced Partition
def data_partition_1(train_items,k,T):

    with open(args.data_path + args.dataset + '/user_pretrain.pk', 'rb') as f:
        uidW = pickle.load(f)
    with open(args.data_path + args.dataset + '/item_pretrain.pk', 'rb') as f:
        iidW = pickle.load(f)

    # get_data_interactions_1
    data = []
    for i in train_items:
        for j in train_items[i]:
            data.append([i, j])

    # Randomly select k centroids
    max_data = 1.2 * len(data) / k
    centroids = random.sample(data, k)


    # centro emb

    centroembs = []
    for i in range(k):
        temp_u = uidW[centroids[i][0]]
        temp_i = iidW[centroids[i][1]]
        centroembs.append([temp_u, temp_i])


    for _ in range(T):
        C = [{} for i in range(k)]
        C_num=[0 for i in range(k)]
        Scores = {}
        for i in range(len(data)):
            for j in range(k):

                score_u = E_score2(uidW[data[i][0]],centroembs[j][0])
                score_i = E_score2(iidW[data[i][1]],centroembs[j][1])

                
                Scores[i, j] = -score_u * score_i


        Scores = sorted(Scores.items(), key=lambda x: x[1], reverse=True)

        fl = set()
        for i in range(len(Scores)):
            if Scores[i][0][0] not in fl:

                if C_num[Scores[i][0][1]] < max_data:
                    if data[Scores[i][0][0]][0] not in C[Scores[i][0][1]]:
                        C[Scores[i][0][1]][data[Scores[i][0][0]][0]]=[data[Scores[i][0][0]][1]]
                    else:
                        C[Scores[i][0][1]][data[Scores[i][0][0]][0]].append(data[Scores[i][0][0]][1])
                    fl.add(Scores[i][0][0])
                    C_num[Scores[i][0][1]] +=1

        centroembs_next = []
        for i in range(k):
            temp_u = []
            temp_i = []

            for j in list(C[i].keys()):
                for l in C[i][j]:
                    temp_u.append(uidW[j])
                    temp_i.append(iidW[l])
            centroembs_next.append([np.mean(temp_u), np.mean(temp_i)])

        loss = 0.0

        for i in range(k):
            score_u = E_score2(centroembs_next[i][0],centroembs[i][0])

            score_i = E_score2(centroembs_next[i][1],centroembs[i][1])

            loss += (score_u * score_i)

        centroembs = centroembs_next
        for i in range(k):
            print(C_num[i])

        print(_, loss)

    users = [[] for i in range(k)]
    items = [[] for i in range(k)]

    for i in range(k):
        users[i] = list(C[i].keys())
        for j in list(C[i].keys()):
            for l in C[i][j]:
                if l not in items[i]:
                    items[i].append(l)


    return C,users,items


# User-based Balanced Partition
def data_partition_2(train_items,k,T):
    data = train_items

    with open(args.data_path + args.dataset + '/user_pretrain.pk', 'rb') as f:
        uidW = pickle.load(f)

    # get_data_interactions_1

    # Randomly select k centroids
    max_data = 1.2 * len(data) / k
    # print data
    centroids = random.sample(list(data.keys()), k)

    # centro emb
    # print centroids

    centroembs = []
    for i in range(k):
        temp_u = uidW[centroids[i]]
        centroembs.append(temp_u)

    for _ in range(T):
        C = [{} for i in range(k)]
        Scores = {}
        for i in list(data.keys()):
            for j in range(k):
                score_u = E_score2(uidW[i],centroembs[j])

                Scores[i, j] = -score_u

        Scores = sorted(Scores.items(), key=lambda x: x[1], reverse=True)

        fl = set()
        for i in range(len(Scores)):
            if Scores[i][0][0] not in fl:
                if len(C[Scores[i][0][1]]) < max_data:
                    C[Scores[i][0][1]][Scores[i][0][0]] = data[Scores[i][0][0]]
                    fl.add(Scores[i][0][0])

        centroembs_next = []
        for i in range(k):
            temp_u = []
            for u in list(C[i].keys()):
                temp_u.append(uidW[u])
            centroembs_next.append(np.mean(temp_u))

        loss = 0.0

        for i in range(k):
            print(len(C[i]))

        for i in range(k):
            score_u = E_score2(centroembs_next[i],centroembs[i])
            loss += score_u

        centroembs = centroembs_next
        print(_, loss)

    users = [[] for i in range(k)]
    items = [[] for i in range(k)]
    for i in range(k):
        users[i]=list(C[i].keys())
        for j in list(C[i].keys()):
            for l in C[i][j]:
                if l not in items[i]:
                    items[i].append(l)

    return C,users,items


#Random
def data_partition_3(train_items,k):

    data = []
    for i in train_items:
        for j in train_items[i]:
            data.append([i, j])

    index = list(range(len(data)))

    random.shuffle(index)

    elem_num = int(len(data) / k)

    C = [{} for i in range(k)]


    for idx in range(k):
        start = idx*elem_num
        if idx!= k-1:
            end = (idx+1)*elem_num
        else:
            end = len(data)
        for i in index[start:end]:
            if data[i][0] not in C[idx]:
                C[idx][data[i][0]]=[data[i][1]]
            else:
                C[idx][data[i][0]].append(data[i][1])

    users = [[] for i in range(k)]
    items = [[] for i in range(k)]
    for i in range(k):
        users[i]=list(C[i].keys())
        for j in list(C[i].keys()):
            for l in C[i][j]:
                if l not in items[i]:
                    items[i].append(l)

    return C,users,items


# Item-based Balanced Partition
def data_partition_4(train_items,k,T):
    # Transpose: group by items instead of users
    item_to_users = {}
    for u in train_items:
        for i in train_items[u]:
            if i not in item_to_users:
                item_to_users[i] = []
            item_to_users[i].append(u)

    with open(args.data_path + args.dataset + '/item_pretrain.pk', 'rb') as f:
        iidW = pickle.load(f)

    # get_data_interactions (by items)
    data = list(item_to_users.keys())

    # Randomly select k centroids
    max_data = 1.2 * len(data) / k
    centroids = random.sample(data, k)

    # centro emb (item embeddings)
    centroembs = []
    for i in range(k):
        centroembs.append(iidW[centroids[i]])

    for _ in range(T):
        C = [{} for i in range(k)]
        Scores = {}
        for i in data:
            for j in range(k):
                score_i = E_score2(iidW[i], centroembs[j])
                Scores[i, j] = -score_i

        Scores = sorted(Scores.items(), key=lambda x: x[1], reverse=True)

        fl = set()
        for i in range(len(Scores)):
            if Scores[i][0][0] not in fl:
                if len(C[Scores[i][0][1]]) < max_data:
                    C[Scores[i][0][1]][Scores[i][0][0]] = item_to_users[Scores[i][0][0]]
                    fl.add(Scores[i][0][0])

        centroembs_next = []
        for i in range(k):
            temp_i = []
            for item in list(C[i].keys()):
                temp_i.append(iidW[item])
            if len(temp_i) > 0:
                centroembs_next.append(np.mean(temp_i))
            else:
                centroembs_next.append(centroids[i])

        loss = 0.0
        for i in range(k):
            print(len(C[i]))

        for i in range(k):
            score_i = E_score2(centroembs_next[i], centroembs[i])
            loss += score_i

        centroembs = centroembs_next
        print(_, loss)

    # Build users and items lists for each partition
    # Transpose: C is {item: [users]}, but training code expects {user: [items]}
    C_t = [{} for i in range(k)]
    users = [[] for i in range(k)]
    items = [[] for i in range(k)]
    for i in range(k):
        items[i] = list(C[i].keys())
        for item in list(C[i].keys()):
            for u in C[i][item]:
                if u not in C_t[i]:
                    C_t[i][u] = []
                if u in train_items:
                    for it in train_items[u]:
                        if it in C[i]:
                            C_t[i][u].append(it)
                if u not in users[i]:
                    users[i].append(u)
        # Ensure every user has at least one item in this partition
        for u in list(C_t[i].keys()):
            if len(C_t[i][u]) == 0:
                # Find a random item from this partition
                if items[i]:
                    C_t[i][u] = [items[i][0]]

    return C_t, users, items
















