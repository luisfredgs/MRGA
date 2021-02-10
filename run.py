
base_dir = '/data5/d01/RGCN/JAPE/'
folder = base_dir + 'data/dbp15k/ja_en/mtranse/0_3/'

from utils import *
from models import *
from tqdm import tqdm
import gc
import torch
from torch.utils.data import Dataset
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import pickle
import collections

os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


random.seed(1)
np.random.seed(1)
torch.manual_seed(1) #cpu
torch.cuda.manual_seed(1) #gpu

shuffle = False
selflearning = False 

Amode = 'bidirection'
#Amode ='undirection'
#Amode ='direction'
Rmode = 'concat'
RGATmode = 4
Diag = True
wo_K = True

sup_align = True

Preprocessmode = 'ori'
#Preprocessmode = 'ref_cross'
#Preprocessmode = 'all_cross'

LossMode = 'triplet_margin'
#LossMode = 'NCE'
#LossMode = 'transe'
#LossMode = 'align'
#LossMode = 'crosstri'

gamma=1

ref_ent1, ref_ent2 = read_references(folder + 'ref_pairs')
assert len(ref_ent1) == len(ref_ent2)
print("To aligned entities:", len(ref_ent1))
sup_ent1, sup_ent2 = read_references(folder + 'sup_pairs')
assert len(sup_ent1) == len(sup_ent2)
print("Aligned entities:", len(sup_ent1))
SelfLoop=True

triples_set1,triples_set2,ent_n,rel_n,ents1,ents2,rels2 = read_corpus(folder)
rel_loop = rel_n
ori_triples = list(triples_set1) + list(triples_set2)
triples, r_hs, r_ts = preprocess_triples(triples_set1, triples_set2, sup_ent1, sup_ent2,list(ents1)+list(ents2),rel_loop)
if Preprocessmode == 'ref_cross':
    cross_triples = add_cross_edges_ref(triples_set1, triples_set2, sup_ent1, sup_ent2, ref_ent1,ref_ent2 )
    triples = triples + cross_triples
elif Preprocessmode == 'all_cross':
    cross_triples1,cross_triples2 = add_cross_edges2(triples_set1, triples_set2, sup_ent1, sup_ent2)
    ref_cross_triples = add_cross_edges_ref(triples_set1, triples_set2, sup_ent1, sup_ent2, ref_ent1,ref_ent2 )
    triples = triples + cross_triples1 + cross_triples2
else:
    cross_triples1,cross_triples2 = add_cross_edges2(triples_set1, triples_set2, sup_ent1, sup_ent2)
    ref_cross_triples = add_cross_edges_ref(triples_set1, triples_set2, sup_ent1, sup_ent2, ref_ent1,ref_ent2 )    

#sup_align_triples = add_sup_align(sup_ent1, sup_ent2,rel_loop)


if SelfLoop:
    rel_n+=1
if Amode == 'undirection':
    triples = add_undirection(triples)
    re_rel_n = 0
elif Amode == 'bidirection':
    triples = add_bidirection(triples, rel_n)
    re_rel_n = rel_n
    rels2 =torch.cat((torch.LongTensor(rels2),torch.LongTensor(rels2)+rel_n))
else:
    re_rel_n = 0
if SelfLoop:    
    selfloop_triples = add_self_loop(sup_ent1, sup_ent2, list(ents1)+list(ents2),rel_loop)
    triples =list(set(list(triples) + selfloop_triples))

else:
    triples =list(set(list(triples)))               

cross_h_rt1 = generate_h_rt(cross_triples1)
cross_h_rt2= generate_h_rt(cross_triples2)

    
A = torch.LongTensor(triples).transpose(0,1)
r_count = torch.bincount(A[1,:])

epoch_KE, epoch_CG = 0, 0
K_CG = 20
K_KE = 2
dist = 2

input_dim = 128
entity_emb = nn.Embedding(ent_n, input_dim)
#entity_emb = nn.Embedding(ent_n, input_dim).from_pretrained(attention_enhanced_emb)
nn.init.normal_(entity_emb.weight, std=1.0 / math.sqrt(ent_n))
entity_emb.requires_grad = False
entity_emb = entity_emb.to(device)


relation_emb = nn.Embedding(rel_n, input_dim)
nn.init.normal_(relation_emb.weight, std=1.0 / math.sqrt(rel_n))
if wo_K:
    relation_emb.requires_grad = True
else:
    relation_emb.requires_grad = True

relation_emb = relation_emb.to(device)
input_idx = torch.LongTensor(np.arange(ent_n)).to(device)
inputr_idx = torch.LongTensor(np.arange(rel_n)).to(device)


# Set model
n_units = [input_dim ,input_dim ,input_dim]
n_heads = [2,2]
cross_graph_model = RGAT(n_units=n_units, n_heads=n_heads,dropout=0.3, attn_dropout=0, instance_normalization=False,l2norm=True,diag=Diag, Rmode=Rmode,RGATmode=RGATmode).to(device)
#cross_graph_model = LPmodel(n_units=n_units, n_heads=n_heads,dropout=0, attn_dropout=0, instance_normalization=False,l2norm=True,diag=Diag, Rmode=Rmode,RGATmode=RGATmode).to(device)
#params = [{'params': filter(lambda p: p.requires_grad, list(cross_graph_model.parameters()) + [entity_emb.weight])},
          #{'params':[relation_emb.weight],'lr':0.005}]
params = [{'params': filter(lambda p: p.requires_grad, list(cross_graph_model.parameters()) + [entity_emb.weight])},
          {'params':[relation_emb.weight],'lr':0.01}]
optimizer = optim.Adagrad(params, lr=0.005, weight_decay=0)
print(cross_graph_model)

train_ill_ori = np.array(list(zip(sup_ent1,sup_ent2)), dtype=np.int32)
train_ill= train_ill_ori
test_ill= np.array(list(zip(ref_ent1, ref_ent2)), dtype=np.int32)

train_left=train_left_ori = torch.LongTensor((np.ones((train_ill_ori.shape[0], K_CG)) * (train_ill_ori[:, 0].reshape((train_ill_ori.shape[0], 1)))).reshape((train_ill_ori.shape[0] * K_CG,))).to(device)
train_right=train_right_ori = torch.LongTensor((np.ones((train_ill_ori.shape[0], K_CG)) * (train_ill_ori[:, 1].reshape((train_ill_ori.shape[0], 1)))).reshape((train_ill_ori.shape[0] * K_CG,))).to(device)
test_left = torch.LongTensor(test_ill[:, 0].squeeze()).to(device)
test_right = torch.LongTensor(test_ill[:, 1].squeeze()).to(device)
print("\ttrain pos/neg_pairs shape: {}".format(train_left.shape))


if wo_K:
    print("w\\o K")
if wo_NNS :
    print("w\\o wo_NNS ")
aug_k = 5
aug_pairs = np.array([],dtype=np.int32)
neg_sample_random = False
batchsize = 20000
wo_K =True
selflearning=True
A_ori = A
maxh1,maxh10,maxmrr=0,0,0

for epoch in range(0,1500):
    t_epoch = time.time()
    cross_graph_model.train()
    optimizer.zero_grad()

    if Amode == 'bidirection':
        if r_prealign:
            attention_enhanced_emb = cross_graph_model(entity_emb(input_idx),torch.cat((relation_emb(inputr_idx),torch.zeros(input_dim).unsqueeze(0).to(device),-relation_emb(inputr_idx)),dim=0),A,torch.LongTensor(ents2))
        else:
            #attention_enhanced_emb = cross_graph_model(entity_emb(input_idx),torch.cat((relation_emb(inputr_idx),torch.zeros(input_dim).unsqueeze(0).to(device),-relation_emb(inputr_idx)),dim=0),A)
            if RGATmode==3:
                attention_enhanced_emb = cross_graph_model(entity_emb(input_idx),A,r_count.to(device))
            else:
                if SelfLoop:
                    attention_enhanced_emb = cross_graph_model(entity_emb(input_idx),torch.cat((relation_emb(inputr_idx),-relation_emb(inputr_idx)[:-1,:]),dim=0),A,torch.LongTensor(r_count).to(device))
                else:
                    attention_enhanced_emb = cross_graph_model(entity_emb(input_idx),torch.cat((relation_emb(inputr_idx),-relation_emb(inputr_idx)),dim=0),A,torch.LongTensor(ents2))                  
    else:
        attention_enhanced_emb = cross_graph_model(entity_emb(input_idx),relation_emb(inputr_idx),A,torch.LongTensor(r_count).to(device))
    


    if epoch  % 4 == 0:
        if epoch  < 5:
            if neg_sample_random:
                neg_left = torch.LongTensor(np.random.choice(ents2, train_ill_ori.shape[0] * K_CG)).to(device)
                neg_right = torch.LongTensor(np.random.choice(ents1, train_ill_ori.shape[0] * K_CG)).to(device)   
            else:
                with torch.no_grad():
                    #neg_left, neg_right = nearest_neighbor_sampling2(attention_enhanced_emb.cpu(),train_ill[:, 0], train_ill[:, 1],K_CG)
                    #neg_left, neg_right = nearest_neighbor_sampling_d01(attention_enhanced_emb.cpu(), torch.LongTensor(train_ill_ori[:, 0]), torch.LongTensor(train_ill_ori[:, 1]),torch.LongTensor(ents1),torch.LongTensor(ents2),K_CG)
                    neg_left, neg_right = nearest_neighbor_sampling(attention_enhanced_emb.cpu(), torch.LongTensor(train_ill_ori[:, 0]), torch.LongTensor(train_ill_ori[:, 1]),K_CG)
                    neg_left, neg_right = neg_left.to(device), neg_right.to(device)
                    
        else:

            if selflearning:                
                #aug_pairs = generate_candidate(attention_enhanced_emb,test_left,test_right,aug_pairs,aug_k,threshold)
                aug_pairs = greedy_align(attention_enhanced_emb,test_left,test_right)
                #A_update = update_A(triples_set1, triples_set2,aug_pairs )
                #A = torch.cat((A_ori,A_update ),dim=1)
                if aug_pairs.shape[0]==0:
                    train_ill= train_ill_ori
                else:
                    train_ill = np.concatenate((train_ill_ori,aug_pairs), axis=0)

                train_left = torch.LongTensor((np.ones((train_ill.shape[0], K_CG)) * (train_ill[:, 0].reshape((train_ill.shape[0], 1)))).reshape((train_ill.shape[0] * K_CG,))).to(device)
                train_right = torch.LongTensor((np.ones((train_ill.shape[0], K_CG)) * (train_ill[:, 1].reshape((train_ill.shape[0], 1)))).reshape((train_ill.shape[0] * K_CG,))).to(device)
            else:
                train_ill= train_ill_ori

            with torch.no_grad():
                neg_left, neg_right = nearest_neighbor_sampling(attention_enhanced_emb.cpu(), torch.LongTensor(train_ill[:, 0]), torch.LongTensor(train_ill[:, 1]),K_CG)
                neg_left, neg_right = neg_left.to(device), neg_right.to(device)



    epoch_CG += 1
    # Cross-graph model alignment loss

    if LossMode == 'align':
        loss_CG = torch.mean(torch.norm(attention_enhanced_emb[train_left]-attention_enhanced_emb[train_right],dim=1)+
                                F.relu(gamma-torch.norm(attention_enhanced_emb[train_left]-attention_enhanced_emb[neg_left],dim=1))+
                                F.relu(gamma-torch.norm(attention_enhanced_emb[train_right]-attention_enhanced_emb[neg_right],dim=1)))
    else:
        loss_CG = F.triplet_margin_loss(torch.cat((attention_enhanced_emb[train_left], attention_enhanced_emb[train_right]), dim=0),
                                        torch.cat((attention_enhanced_emb[train_right], attention_enhanced_emb[train_left]), dim=0),
                                        torch.cat((attention_enhanced_emb[neg_left], attention_enhanced_emb[neg_right]), dim=0),
                                        margin=2, p=2)
            
    loss_CG.backward()
    optimizer.step()  
    '''
    with torch.no_grad():
        inputr = torch.zeros((r_count.shape[0],input_dim)).to(device)
        edge_h = attention_enhanced_emb[A[2, :], :]- attention_enhanced_emb[A[0, :], :]
        inputr.index_add_(0,A[1, :].cuda(),edge_h)
        inputr = inputr.div(r_count.to(device).unsqueeze(1))
        del edge_h
    '''
    print("loss_CG in epoch {:d}: {:f}, time: {:.4f} s".format(epoch, loss_CG.item(), time.time() - t_epoch))
    print("loss_true in epoch {:d}: {:f}, time: {:.4f} s".format(epoch, torch.mean(torch.norm(attention_enhanced_emb[train_left]-attention_enhanced_emb[train_right],dim=1)).item(), time.time() - t_epoch))

    
    # Test
    if (epoch + 1) % 10 == 0:
        print("\nepoch {:d}, checkpoint!".format(epoch))

        with torch.no_grad():
            t_test = time.time()
            cross_graph_model.eval()
            
            if Amode == 'bidirection':
                if r_prealign:
                    attention_enhanced_emb = cross_graph_model(entity_emb(input_idx),torch.cat((relation_emb(inputr_idx),torch.zeros(input_dim).unsqueeze(0).to(device),-relation_emb(inputr_idx)),dim=0),A)
                else:
                    #attention_enhanced_emb = cross_graph_model(entity_emb(input_idx),torch.cat((relation_emb(inputr_idx),torch.zeros(input_dim).unsqueeze(0).to(device),-relation_emb(inputr_idx)),dim=0),A)
                    if RGATmode==3:
                        attention_enhanced_emb = cross_graph_model(entity_emb(input_idx),A,r_count.to(device))
                    else:
                        if SelfLoop:
                            #LP
                            #attention_enhanced_emb = cross_graph_model(entity_emb(input_idx),torch.cat((relation_emb(inputr_idx),-relation_emb(inputr_idx)[:-1,:]),dim=0),A,cfdc.to(device),torch.LongTensor(sup_ent1+sup_ent2).to(device),True)
                            #attention_enhanced_emb = cross_graph_model(entity_emb(input_idx),torch.cat((relation_emb(inputr_idx),torch.zeros(input_dim).unsqueeze(0).to(device),-relation_emb(inputr_idx)),dim=0),A,cfdc.to(device),torch.LongTensor(sup_ent1+sup_ent2).to(device),True)
                            #2   
                            #attention_enhanced_emb = cross_graph_model(entity_emb(input_idx),relation_emb(inputr_idx),A)
                            attention_enhanced_emb = cross_graph_model(entity_emb(input_idx),torch.cat((relation_emb(inputr_idx),-relation_emb(inputr_idx)[:-1,:]),dim=0),A,torch.LongTensor(r_count).to(device))
                            #attention_enhanced_emb = cross_graph_model(entity_emb(input_idx),torch.cat((relation_emb(inputr_idx),-relation_emb(inputr_idx)[:-1,:]),dim=0),A_ori,torch.LongTensor(r_count).to(device))
                        else:
                            attention_enhanced_emb = cross_graph_model(entity_emb(input_idx),torch.cat((relation_emb(inputr_idx),-relation_emb(inputr_idx)),dim=0),A,torch.LongTensor(ents2))
            else:
                attention_enhanced_emb = cross_graph_model(entity_emb(input_idx),relation_emb(inputr_idx),A,torch.LongTensor(r_count).to(device))
                
            top_k = [1, 5, 10, 50, 100]
            
            print('test_left')
            h1 = greedy_align(attention_enhanced_emb,test_left,test_right,rounds=3,test=True)
            print(h1)
            if h1>maxh1:
                maxh1=h1
            h1,h10,mrr = model_test(top_k,pairwise_distances(attention_enhanced_emb[test_left], attention_enhanced_emb[test_right]))
            if h1>maxh1:
                maxh1=h1
            if h10>maxh10:
                maxh10=h10
            if mrr>maxmrr:
                maxmrr=mrr
            print('test_right')
            h1,h10,mrr = model_test(top_k,pairwise_distances(attention_enhanced_emb[test_right], attention_enhanced_emb[test_left]))
            if h1>maxh1:
                maxh1=h1
            if h10>maxh10:
                maxh1=h10
            if mrr>maxmrr:
                maxmrr=mrr 
            print(maxh1,maxh10,maxmrr)


            del attention_enhanced_emb
            gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
