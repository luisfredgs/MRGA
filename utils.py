from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import os, time, multiprocessing
import math
import random
import numpy as np
import scipy
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset


def read_corpus(folder):
    triples_set1 = read_triples(folder + 'triples_1')
    triples_set2 = read_triples(folder + 'triples_2')
    heads1 = set([triple[0] for triple in triples_set1])
    props1 = set([triple[1] for triple in triples_set1])
    tails1 = set([triple[2] for triple in triples_set1])
    ents1 = heads1 | tails1

    heads2 = set([triple[0] for triple in triples_set2])
    props2 = set([triple[1] for triple in triples_set2])
    tails2 = set([triple[2] for triple in triples_set2])
    ents2 = heads2 | tails2

    ents = ents1 | ents2
    rels = props1 | props2
    print('entity1:'+str(len(ents1))+'+entity2:'+str(len(ents2))+'all:'+str(len(ents)))
    print('rel1:'+str(len(props1))+'+rel2:'+str(len(props2))+'all:'+str(len(rels)))
    
    ent_n = len(ents)
    rel_n = len(rels)

    
    return triples_set1,triples_set2,ent_n,rel_n,list(ents1),list(ents2),list(props2)

def read_triples(file):
    triples = set()
    with open(file, 'r', encoding='utf8') as f:
        for line in f:
            params = line.strip('\n').split('\t')
            assert len(params) == 3
            h = int(params[0])
            r = int(params[1])
            t = int(params[2])
            triples.add((h, r, t))
        f.close()
    return triples
    
def read_references(file):
    ref1, ref2 = list(), list()
    with open(file, 'r', encoding='utf8') as f:
        for line in f:
            params = line.strip('\n').split('\t')
            assert len(params) == 2
            e1 = int(params[0])
            e2 = int(params[1])
            ref1.append(e1)
            ref2.append(e2)
        f.close()
        assert len(ref1) == len(ref2)
    return ref1, ref2
def add_cross_edges2(kg1, kg2,ents1, ents2):
    print("before add cross:", len(kg1), len(kg2))
    cross_triples1 = []
    cross_triples2 = []
    links1 = dict(zip(ents1, ents2))
    links2 = dict(zip(ents2, ents1))
    for h1, r1, t1 in kg1:
        h2 = links1.get(h1, None)
        t2 = links1.get(t1, None)
        if h2 is not None: 
            cross_triples1.append((h2, r1, t1))
        if t2 is not None:
            cross_triples1.append((h1, r1, t2))
    for h2, r2, t2 in kg2:
        h1 = links2.get(h2, None)
        t1 = links2.get(t2, None)
        if h1 is not None: 
            cross_triples2.append((h1, r2, t2))
        if t1 is not None:
            cross_triples2.append((h2, r2, t1))
    return cross_triples1,cross_triples2

def generate_cfdc(triples):
    A = np.array(triples).T
    edge1 = list(zip(A[0,:],A[1,:]))
    trCount1 = collections.Counter(edge1)
    confidence1 = []
    for ed in edge1:
        confidence1.append(trCount1[ed])
    cfdc1 = torch.Tensor(confidence1)
    cfdc = torch.div(1,torch.sqrt(cfdc1))
    
    edge2 = list(zip(A[2,:],A[1,:]))
    trCount2 = collections.Counter(edge2)    
    confidence2 = []
    for ed in edge2:
        confidence2.append(trCount2[ed])
    cfdc2 = torch.Tensor(confidence2)
    return cfdc.mul(torch.div(1,torch.sqrt(cfdc2)))

def add_cross_edges(kg1, kg2,ents1, ents2):
    print("before add cross:", len(kg1), len(kg2))
    cross_triples = []
    links1 = dict(zip(ents1, ents2))
    links2 = dict(zip(ents2, ents1))
    for h1, r1, t1 in kg1:
        h2 = links1.get(h1, None)
        t2 = links1.get(t1, None)
        if h2 is not None: 
            cross_triples.append((h2, r1, t1))
        if t2 is not None:
            cross_triples.append((h1, r1, t2))
    for h2, r2, t2 in kg2:
        h1 = links2.get(h2, None)
        t1 = links2.get(t2, None)
        if h1 is not None: 
            cross_triples.append((h1, r2, t2))
        if t1 is not None:
            cross_triples.append((h2, r2, t1))
    print("after add cross:", len(cross_triples))
    return cross_triples

def add_cross_edges_ref(kg1, kg2,ents1, ents2,ref_ents1,ref_ents2):
    print("before add cross:", len(kg1), len(kg2))
    cross_triples = []
    links1 = dict(zip(ents1, ents2))
    links2 = dict(zip(ents2, ents1))
    for h1, r1, t1 in kg1:
        h2 = links1.get(h1, None)
        t2 = links1.get(t1, None)
        if h2 is not None and t1 in ref_ents1: 
            cross_triples.append((h2, r1, t1))
        if t2 is not None and h1 in ref_ents1:
            cross_triples.append((h1, r1, t2))
    for h2, r2, t2 in kg2:
        h1 = links2.get(h2, None)
        t1 = links2.get(t2, None)
        if h1 is not None and t2 in ref_ents2: 
            cross_triples.append((h1, r2, t2))
        if t1 is not None and h2 in ref_ents2:
            cross_triples.append((h2, r2, t1))
    print("after add cross:", len(cross_triples))
    return cross_triples

def enhance_triples_2(kg1, kg2, ents1, ents2):
    assert len(ents1) == len(ents2)
    print("before enhanced:", len(kg1), len(kg2))
    enhanced_triples1, enhanced_triples2 = [], []
    links1 = dict(zip(ents1, ents2))
    links2 = dict(zip(ents2, ents1))
    for h1, r1, t1 in kg1:
        h2 = links1.get(h1, None)
        t2 = links1.get(t1, None)
        if h2 is not None and t2 is not None:
            enhanced_triples2.append((h2, r1, t2))
    for h2, r2, t2 in kg2:
        h1 = links2.get(h2, None)
        t1 = links2.get(t2, None)
        if h1 is not None and t1 is not None:
            enhanced_triples1.append((h1, r2, t1))
    print("after enhanced:", len(enhanced_triples1), len(enhanced_triples2))
    return enhanced_triples1, enhanced_triples2

def add_undirection(triples):
    retriples = []
    for h,r,t in triples:
        retriples.append((t,r,h))
    return list(triples) + retriples

def add_bidirection(triples,rel_n):
    retriples = []
    for h,r,t in triples:
        retriples.append((t,r + rel_n,h))
    return list(triples) + retriples

# 不需要再关于r对称
def add_sup_align(sup_ent1, sup_ent2,rel_loop):
    sup_align = []
    links = dict(zip(sup_ent1+sup_ent2, sup_ent2+sup_ent1))
    for e in links.keys():
        e2 =  links.get(e,None)
        if e2 is not None:
            sup_align.append((e,rel_loop,e2))  
    return sup_align

# 不需要再关于r对称
def add_self_loop(sup_ent1, sup_ent2, ents,rel_loop):
    selfloop = []
    if sup_align:
        links = dict(zip(sup_ent1+sup_ent2, sup_ent2+sup_ent1))
    for e in ents:
        selfloop.append((e,rel_loop,e))
    return selfloop    
    
def preprocess_triples(triples_set1, triples_set2, sup_ent1, sup_ent2,ents,rel_loop):

    def generate_r_ht(triples):
        r_hs, r_ts = {}, {}
        for (h, r, t) in triples:
            if r not in r_hs:
                r_hs[r] = set()
            if r not in r_ts:
                r_ts[r] = set()
            r_hs[r].add(h)
            r_ts[r].add(t)
        assert len(r_hs) == len(r_ts)
        return r_hs, r_ts
    

    enhanced_triples1, enhanced_triples2 = enhance_triples_2(triples_set1, triples_set2, sup_ent1, sup_ent2)
    triples = set(enhanced_triples1+enhanced_triples2+list(triples_set1)+list(triples_set2))
    #triples =  set(list(triples_set1)+list(triples_set2))
    r_hs, r_ts = generate_r_ht(triples)
    print(len(triples)) 
    return list(triples),r_hs, r_ts 

def generate_h_rt(triples):
    h_rt_dict = {}
    for h,r,t in triples:
        h_rt_dict[h] = h_rt_dict.get(h,[])
        h_rt_dict[h].append((h,r,t))      
    return h_rt_dict

def generate_candidate(attention_enhanced_emb,test_left,test_right,aug_pairs,k,threshold):
    neighbors_left = []
    neighbors_right = []
    with torch.no_grad():
        if aug_pairs.shape[0]==0:
            aug_pairs=set()
        else:
            aug_pairs = set(zip(aug_pairs[:,0],aug_pairs[:,1]))
        sim_mat = pairwise_distances(attention_enhanced_emb[test_left], attention_enhanced_emb[test_right]).cpu().numpy()    
        for i in range(sim_mat.shape[0]):
            sort_index_left = np.argpartition(sim_mat[i, :], k)
            sort_index_right = np.argpartition(sim_mat[:, i], k)
            left_pairs = [j for j in itertools.product([i], sort_index_left[0:k])]
            right_pairs = [j for j in itertools.product(sort_index_right[0:k],[i])]
            neighbors_left.extend(left_pairs )
            neighbors_right.extend(right_pairs)
        threshold_neighbors = filter_mat(sim_mat, threshold, False,False)
        if len(threshold_neighbors)==0:
            print('threshold 0 ')
            return np.array([],dtype=np.int32)
        select_pairs =  mwgm_igraph(set(neighbors_left) & set(neighbors_right) & threshold_neighbors , threshold-sim_mat)
        if len(select_pairs)==0:
            print('0/0')
            return np.array([],dtype=np.int32)
        print('augmentation pairs and acc:')
        print(check_align(select_pairs))
        select_pairs = mwgm_igraph(select_pairs | aug_pairs, threshold-sim_mat)
        print(check_align(select_pairs))
    return np.array(list(select_pairs),dtype=np.int32)
            
def filter_mat(mat, threshold, greater=True, equal=False):
    if greater and equal:
        x, y = np.where(mat >= threshold)
    elif greater and not equal:
        x, y = np.where(mat > threshold)
    elif not greater and equal:
        x, y = np.where(mat <= threshold)
    else:
        x, y = np.where(mat < threshold)
    return set(zip(x, y))

def check_align(neighbors):
    corr=0
    for i,j in neighbors:
        if i==j:
            corr+=1
    print(len(neighbors))
    print(corr)
    return  corr/len(neighbors),len(neighbors),corr

def greedy_align(attention_enhanced_emb,test_left,test_right,rounds=1,test=False):
    pairs=[]
    left=test_left.cpu().numpy()
    right=test_right.cpu().numpy()
    k=1
    step=0
    keepgoing=True
    with torch.no_grad():
        sim_mat = pairwise_distances(attention_enhanced_emb[test_left], attention_enhanced_emb[test_right]).cpu().numpy()
        while keepgoing==True and step<rounds:
            neighbors_left=[]
            neighbors_right=[]
            for i in left[np.where(left>=0)]:
                sort_index_left = np.argpartition(sim_mat[i, :], k)
                left_pairs = [j for j in itertools.product([i], sort_index_left[0:k])]
                neighbors_left.extend(left_pairs )
            for i in left[np.where(right>=0)]:
                sort_index_right = np.argpartition(sim_mat[:, i], k)
                right_pairs = [j for j in itertools.product(sort_index_right[0:k],[i])]
                neighbors_right.extend(right_pairs)
            candi = list(set(neighbors_left)& set(neighbors_right))
            pairs.extend(candi )
            if len(candi)<=100:
                keepgoing=False
            index1 = np.array(list(set([pair[0] for pair in pairs])))
            left[index1]=-1
            index2 =  np.array(list(set([pair[1] for pair in pairs])))
            right[index2]=-1
            sim_mat[index1,:]=sim_mat.max()    
            sim_mat[:,index2]=sim_mat.max()
            step=step+1
        if test:
            sim_mat = pairwise_distances(attention_enhanced_emb[test_left], attention_enhanced_emb[test_right]).cpu().numpy()
            for i in left[np.where(left>=0)]:
                sort_index_left = np.argpartition(sim_mat[i, :], 1)
                left_pairs = [j for j in itertools.product([i], sort_index_left[0:k])]
                pairs.extend(left_pairs)
            return check_align(set(pairs))[0]
        
        else:
            print(check_align(set(pairs)))
            return np.array(list(set(pairs)))+np.array([[0,10500]])

def model_test(top_k,distance):
    acc_l2r = np.zeros((len(top_k)), dtype=np.float32)
    #acc_r2l = np.zeros((len(top_k)), dtype=np.float32)
    test_total, test_loss, mean_l2r, mrr_l2r = 0, 0., 0., 0.
    
    for idx in range(distance.shape[0]):
        values, indices = torch.sort(distance[idx, :], descending=False)
        rank = (indices == idx).nonzero().squeeze().item()
        mean_l2r += (rank + 1)
        mrr_l2r += 1.0 / (rank + 1)
        for i in range(len(top_k)):
            if rank < top_k[i]:
                acc_l2r[i] += 1
            #else:
                #if i == len(top_k)-1:
                    #print(name_dict[ref_ent1[idx]],name_dict[ref_ent2[idx]])
    mean_l2r /= distance.size(0)
    mrr_l2r /= distance.size(0)

    for i in range(len(top_k)):
        acc_l2r[i] = round(acc_l2r[i] / distance.shape[0], 4)    
    del distance
    gc.collect()
    print("l2r: acc of top {} = {}, mr = {:.3f}, mrr = {:.3f}, time = {:.4f} s ".format(top_k, acc_l2r, mean_l2r, mrr_l2r, time.time() - t_test))
    return acc_l2r[0],acc_l2r[2],mrr_l2r

def nearest_neighbor_sampling(emb, left, right, K,mode=0,start=1):
    t = time.time()
    if mode ==1:
        K=2*K   
    neg_left = []    
    distance = pairwise_distances(emb[right], emb[right])
    for idx in range(right.shape[0]):
        _, indices = torch.sort(distance[idx, :], descending=False)
        neg_left.append(right[indices[start : K+start]])
    if mode ==1:
        neg_left = np.random.choice(neg_left,K//2)
    neg_left = torch.cat(tuple(neg_left), dim=0)
    neg_right = []
    distance = pairwise_distances(emb[left], emb[left])
    for idx in range(left.shape[0]):
        _, indices = torch.sort(distance[idx, :], descending=False)
        neg_right.append(left[indices[start: K+start]])
    if mode ==1:
        neg_right = np.random.choice(neg_right,K//2)
    neg_right = torch.cat(tuple(neg_right), dim=0)
    return neg_left, neg_right

def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    return torch.clamp(dist, 0.0, np.inf)