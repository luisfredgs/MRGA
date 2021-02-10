#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  File name:    layers.py
  Author:       locke
  Date created: 2018/10/5 下午2:41
"""

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
import numpy as np
import gc

class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b

class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class MultiHeadGraphAttention(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    https://github.com/Diego999/pyGAT/blob/master/layers.py
    """
    def __init__(self, n_head, f_in, f_out, attn_dropout, diag=True, init=None, bias=False):
        super(MultiHeadGraphAttention, self).__init__()
        self.n_head = n_head
        self.f_in = f_in
        self.f_out = f_out
        self.diag = diag
        if self.diag:
            self.w = Parameter(torch.Tensor(n_head , 1, f_out))
        else:
            self.w = Parameter(torch.Tensor(n_head , f_in, f_out))  
        self.a_src_dst = Parameter(torch.Tensor(n_head, f_out * 2, 1))
        self.attn_dropout = attn_dropout
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.special_spmm = SpecialSpmm()
        if bias:
            self.bias = Parameter(torch.Tensor(f_out))
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter('bias', None)
        if init is not None and diag:
            init(self.w)
            stdv = 1. / math.sqrt(self.a_src_dst.size(1))
            nn.init.uniform_(self.a_src_dst, -stdv, stdv)
        else:
            nn.init.xavier_uniform_(self.w)
            nn.init.xavier_uniform_(self.a_src_dst)

    def forward(self, input, adj):
        output = []
        for i in range(self.n_head):
            N = input.size()[0]
            edge = adj._indices()
            

            if self.diag:
                h = torch.mul(input, self.w[i])
            else:
                h = torch.mm(input, self.w[i])

            edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1) # edge: 2*D x E
            edge_e = torch.exp(-self.leaky_relu(edge_h.mm(self.a_src_dst[i]).squeeze())) # edge_e: 1 x E
            
            e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1)).cuda()  if next(self.parameters()).is_cuda else torch.ones(size=(N, 1))) # e_rowsum: N x 1
            edge_e = F.dropout(edge_e, self.attn_dropout, training=self.training)   # edge_e: 1 x E
            
            h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
            h_prime = h_prime.div(e_rowsum)
            output.append(h_prime.unsqueeze(0))
        
        output = torch.cat(output, dim=0)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
    
    def __repr__(self):
        if self.diag:
            return self.__class__.__name__ + ' (' + str(self.f_out) + ' -> ' + str(self.f_out) + ') * ' + str(self.n_head) + ' heads'
        else:
            return self.__class__.__name__ + ' (' + str(self.f_in) + ' -> ' + str(self.f_out) + ') * ' + str(self.n_head) + ' heads'

class RMultiHeadGraphAttention(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    https://github.com/Diego999/pyGAT/blob/master/layers.py
    """
    def __init__(self, n_head, f_in, f_out, attn_dropout, diag=True, init=None, bias=False,Rmode='concat'):
        super(RMultiHeadGraphAttention, self).__init__()
        self.n_head = n_head
        self.f_in = f_in
        self.f_out = f_out
        self.diag = diag
        self.Rmode = Rmode
        if self.diag:
            self.w = Parameter(torch.Tensor(n_head-1, 1, f_out)) #修
        else:
            self.w = Parameter(torch.Tensor(n_head-1, f_in, f_out)) #修
            
        if self.Rmode =='concat':
            self.a_src_dst = Parameter(torch.Tensor(n_head, 3, f_out, 1))
        else:
            self.a_src_dst = Parameter(torch.Tensor(n_head, 2, f_out, 1))
            
        self.attn_dropout = attn_dropout
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.special_spmm = SpecialSpmm()
        if bias:
            self.bias = Parameter(torch.Tensor(f_out))
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter('bias', None)
        if init is not None and diag:
            init(self.w)
            stdv = 1. / math.sqrt(self.a_src_dst.size(1))
            nn.init.uniform_(self.a_src_dst, -stdv, stdv)
        else:
            nn.init.xavier_uniform_(self.w)
            nn.init.xavier_uniform_(self.a_src_dst)

    def forward(self, input, inputr, A, relational=True):
        output = []
        for i in range(self.n_head):
            N = input.size()[0]
            if i<self.n_head-1:   #修
                if self.diag:
                    h = torch.mul(input, self.w[i])
                else:
                    h = torch.mm(input, self.w[i])

            
            if self.Rmode == 'concat':
                edge_h = h[A[0, :], :].mm(self.a_src_dst[i,0])
                edge_h = edge_h + h[A[2, :], :].mm(self.a_src_dst[i,1])
                edge_h = edge_h + h[inputr[1, :], :].mm(self.a_src_dst[i,2])
                #print(edge_h.shape)
                #edge_h = torch.cat((h[A[0, :], :], h[A[2, :], :], inputr[A[1, :], :]),dim=1) # edge: 3*D x E
            elif self.Rmode == 'transe':
                edge_h = torch.cat((h[A[0, :], :], h[A[2, :], :] - inputr[A[1, :], :]),dim=1) # edge: 3*D x E
                edge_h = edge_h.mm(self.a_src_dst[i])
            else:
                edge_h = torch.cat((h[A[0, :], :], h[A[2, :], :]),dim=1)
                edge_h = edge_h.mm(self.a_src_dst[i])

            edge_e = torch.exp(-self.leaky_relu(edge_h.squeeze())) # edge_e: 1 x E
            e_rowsum = self.special_spmm(A[[0,2],:], edge_e, torch.Size([N, N]), torch.ones(size=(N, 1)).cuda()  if next(self.parameters()).is_cuda else torch.ones(size=(N, 1))) # e_rowsum: N x 1
            edge_e = F.dropout(edge_e, self.attn_dropout, training=self.training)   # edge_e: 1 x E
            
            h_prime = self.special_spmm(A[[0,2],:], edge_e, torch.Size([N, N]), h)
            h_prime = h_prime.div(e_rowsum)
            
            output.append(h_prime.unsqueeze(0))
        
        output = torch.cat(output, dim=0)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
    
    def __repr__(self):
        if self.diag:
            return self.__class__.__name__ + ' (' + str(self.f_out) + ' -> ' + str(self.f_out) + ') * ' + str(self.n_head) + ' heads'
        else:
            return self.__class__.__name__ + ' (' + str(self.f_in) + ' -> ' + str(self.f_out) + ') * ' + str(self.n_head) + ' heads'

class RMultiHeadGraphAttention2(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    https://github.com/Diego999/pyGAT/blob/master/layers.py
    """
    def __init__(self, n_head, f_in, f_out, attn_dropout, diag=True, init=None, bias=False,Rmode=''):
        super(RMultiHeadGraphAttention2, self).__init__()
        self.n_head = n_head
        self.f_in = f_in
        self.f_out = f_out
        self.diag = diag
        self.Rmode = Rmode
        
        if Rmode == 'mapping':
            #self.Map = Parameter(torch.Tensor(n_head-1, f_out, f_out))
            #nn.init.orthogonal(self.Map)
            self.Map_ori = Parameter(torch.Tensor(1,f_out))
   
        if self.diag:
            self.w = Parameter(torch.Tensor(n_head-1, 1, f_out)) #修
        else:
            self.w_ori = Parameter(torch.Tensor(1,f_out)) #修
            
        self.a_src_dst = Parameter(torch.Tensor(n_head, 2, f_out, 1))
            
        self.attn_dropout = attn_dropout
        self.leaky_relu = nn.LeakyReLU(negative_slope = 0.2)
        self.special_spmm = SpecialSpmm()
        if bias:
            self.bias = Parameter(torch.Tensor(f_out))
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter('bias', None)
        if init is not None and diag:
            init(self.w)
            stdv = 1. / math.sqrt(self.a_src_dst.size(1)*self.a_src_dst.size(2))
            #nn.init.uniform_(self.a_src_dst, -stdv, stdv)
            nn.init.normal_(self.a_src_dst, mean=0, std=stdv)
        else:
            #nn.init.xavier_uniform_(self.w)
            #nn.init.xavier_uniform_(self.a_src_dst)
            nn.init.normal_(self.w_ori,std=1.0/math.sqrt(f_out))
            #nn.init.normal_(self.a_src_dst, mean=1.0, std=1.0)
            stdv = 1. / math.sqrt(self.a_src_dst.size(1)*self.a_src_dst.size(2))
            #nn.init.uniform_(self.a_src_dst, -stdv, stdv)
            nn.init.normal_(self.a_src_dst, mean=0, std=stdv)  


            
    def forward(self, h, inputr, A,ents2=None, relational=True):
        output = []
        N = h.size()
        for i in range(self.n_head):

            if i>=1:   #修
                if self.diag:
                    h = torch.mul(h, self.w[i-1])
                    if self.Rmode == 'mapping':
                        Map= F.normalize(self.Map_ori,dim=1)
                        Map = torch.eye(Map.size(-1)).cuda()-2*Map.T.mm(Map)
                        h[ents2] = h[ents2].mm(Map)
                        
                    #inputr = torch.mul(inputr, self.w[i-1])
                else:
                    w= F.normalize(self.w_ori,dim=1)
                    w = torch.eye(w.size(-1)).cuda()-2*w.T.mm(w)
                    h = torch.mm(h,w)
            '''
            if i>=1: 
                edge_h = torch.mul(h, self.w[i-1])[A[2, :], :].mm(self.a_src_dst[i,0])
            else:
                edge_h = h[A[2, :], :].mm(self.a_src_dst[i,0])
            '''
            edge_h = h[A[2, :], :].mm(self.a_src_dst[i,0])
            edge_h = edge_h + inputr[A[1, :], :].mm(self.a_src_dst[i,1])
            edge_e = torch.exp(-self.leaky_relu(edge_h.squeeze())) # edge_e: 1 x 
            edge_e = F.dropout(edge_e, self.attn_dropout, training=self.training)   # edge_e: 1 x E            
            #e_rowsum = self.special_spmm(A[[0,2],:], edge_e, torch.Size([N[0], N[0]]), torch.ones(size=(N[0], 1)).cuda()  if next(self.parameters()).is_cuda else torch.ones(size=(N[0], 1))) # e_rowsum: N x 1
            e_rowsum = torch.sparse.FloatTensor(A[[0],:].cuda(), edge_e).to_dense().unsqueeze(1)
            edge_h = h[A[2, :], :] - inputr[A[1, :], :]
            if i ==self.n_head-1:
                del inputr
                del h
                gc.collect()
            edge_h = torch.mul(edge_h,edge_e.unsqueeze(0).transpose(0,1))        
            h_prime = torch.zeros(N).cuda()
            h_prime.index_add_(0,A[0, :].cuda(),edge_h)         
            h_prime = h_prime.div(e_rowsum)
            output.append(h_prime.unsqueeze(0))
            del h_prime
            del edge_h
            del edge_e
            del e_rowsum
            gc.collect()
        del A
        gc.collect()   
        
        output = torch.cat(output, dim=0)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
    
    def __repr__(self):
        if self.diag:
            return self.__class__.__name__ + ' (' + str(self.f_out) + ' -> ' + str(self.f_out) + ') * ' + str(self.n_head) + ' heads'
        else:
            return self.__class__.__name__ + ' (' + str(self.f_in) + ' -> ' + str(self.f_out) + ') * ' + str(self.n_head) + ' heads'

class RMultiHeadGraphAttentionab(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    https://github.com/Diego999/pyGAT/blob/master/layers.py
    """
    def __init__(self, n_head, f_in, f_out, attn_dropout, diag=True, init=None, bias=False,Rmode=''):
        super(RMultiHeadGraphAttentionab, self).__init__()
        self.n_head = n_head
        self.f_in = f_in
        self.f_out = f_out
        self.diag = diag
        self.Rmode = Rmode
        
        if Rmode == 'mapping':
            #self.Map = Parameter(torch.Tensor(n_head-1, f_out, f_out))
            #nn.init.orthogonal(self.Map)
            self.Map_ori = Parameter(torch.Tensor(1,f_out))
   
        if self.diag:
            self.w = Parameter(torch.Tensor(n_head-1, 1, f_out)) #修
        else:
            self.w_ori = Parameter(torch.Tensor(1,f_out)) #修
            
        self.a_src_dst = Parameter(torch.Tensor(n_head, 2, f_out, 1))
            
        self.attn_dropout = attn_dropout
        self.leaky_relu = nn.LeakyReLU(negative_slope = 0.2)
        self.special_spmm = SpecialSpmm()
        if bias:
            self.bias = Parameter(torch.Tensor(f_out))
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter('bias', None)
        if init is not None and diag:
            init(self.w)
            stdv = 1. / math.sqrt(self.a_src_dst.size(1)*self.a_src_dst.size(2))
            #nn.init.uniform_(self.a_src_dst, -stdv, stdv)
            nn.init.normal_(self.a_src_dst, mean=0, std=stdv)
        else:
            #nn.init.xavier_uniform_(self.w)
            #nn.init.xavier_uniform_(self.a_src_dst)
            nn.init.normal_(self.w_ori,std=1.0/math.sqrt(f_out))
            #nn.init.normal_(self.a_src_dst, mean=1.0, std=1.0)
            stdv = 1. / math.sqrt(self.a_src_dst.size(1)*self.a_src_dst.size(2))
            #nn.init.uniform_(self.a_src_dst, -stdv, stdv)
            nn.init.normal_(self.a_src_dst, mean=0, std=stdv)  


            
    def forward(self, h, inputr, A,ents2=None, relational=True):
        output = []
        N = h.size()
        for i in range(self.n_head):

            if i>=1:   #修
                if self.diag:
                    h = torch.mul(h, self.w[i-1])
                    if self.Rmode == 'mapping':
                        Map= F.normalize(self.Map_ori,dim=1)
                        Map = torch.eye(Map.size(-1)).cuda()-2*Map.T.mm(Map)
                        h[ents2] = h[ents2].mm(Map)
                        
                    #inputr = torch.mul(inputr, self.w[i-1])
                else:
                    w= F.normalize(self.w_ori,dim=1)
                    w = torch.eye(w.size(-1)).cuda()-2*w.T.mm(w)
                    h = torch.mm(h,w)
            '''
            if i>=1: 
                edge_h = torch.mul(h, self.w[i-1])[A[2, :], :].mm(self.a_src_dst[i,0])
            else:
                edge_h = h[A[2, :], :].mm(self.a_src_dst[i,0])
            '''
            edge_h = h[A[2, :], :].mm(self.a_src_dst[i,0])
            #edge_h = edge_h + inputr[A[1, :], :].mm(self.a_src_dst[i,1])
            edge_e = torch.exp(-self.leaky_relu(edge_h.squeeze())) # edge_e: 1 x 
            edge_e = F.dropout(edge_e, self.attn_dropout, training=self.training)   # edge_e: 1 x E            
            #e_rowsum = self.special_spmm(A[[0,2],:], edge_e, torch.Size([N[0], N[0]]), torch.ones(size=(N[0], 1)).cuda()  if next(self.parameters()).is_cuda else torch.ones(size=(N[0], 1))) # e_rowsum: N x 1
            e_rowsum = torch.sparse.FloatTensor(A[[0],:].cuda(), edge_e).to_dense().unsqueeze(1)
            edge_h = h[A[2, :], :]
            if i ==self.n_head-1:
                del inputr
                del h
                gc.collect()
            edge_h = torch.mul(edge_h,edge_e.unsqueeze(0).transpose(0,1))        
            h_prime = torch.zeros(N).cuda()
            h_prime.index_add_(0,A[0, :].cuda(),edge_h)         
            h_prime = h_prime.div(e_rowsum)
            output.append(h_prime.unsqueeze(0))
            del h_prime
            del edge_h
            del edge_e
            del e_rowsum
            gc.collect()
        del A
        gc.collect()   
        
        output = torch.cat(output, dim=0)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
    
    def __repr__(self):
        if self.diag:
            return self.__class__.__name__ + ' (' + str(self.f_out) + ' -> ' + str(self.f_out) + ') * ' + str(self.n_head) + ' heads'
        else:
            return self.__class__.__name__ + ' (' + str(self.f_in) + ' -> ' + str(self.f_out) + ') * ' + str(self.n_head) + ' heads'

class RMultiHeadGraphAttention2m(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    https://github.com/Diego999/pyGAT/blob/master/layers.py
    """
    def __init__(self, n_head, f_in, f_out, attn_dropout, diag=True, init=None, bias=False,Rmode='',device='cuda:0'):
        super(RMultiHeadGraphAttention2m, self).__init__()
        self.n_head = n_head
        self.f_in = f_in
        self.f_out = f_out
        self.diag = diag
        self.Rmode = Rmode
        self.device = device
        
        if self.diag:
            self.w = Parameter(torch.Tensor(n_head-1, 1, f_out)) #修
        else:
            self.w_ori = Parameter(torch.Tensor(1,f_out)) #修
            
        self.a_src_dst = Parameter(torch.Tensor(n_head, 2, f_out, 1))
            
        self.attn_dropout = attn_dropout
        self.leaky_relu = nn.LeakyReLU(negative_slope = 0.2)
        self.special_spmm = SpecialSpmm()
        if bias:
            self.bias = Parameter(torch.Tensor(f_out))
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter('bias', None)
        if init is not None and diag:
            init(self.w)
            stdv = 1. / math.sqrt(self.a_src_dst.size(1)*self.a_src_dst.size(2))
            #nn.init.uniform_(self.a_src_dst, -stdv, stdv)
            nn.init.normal_(self.a_src_dst, mean=0, std=stdv)
        else:
            #nn.init.xavier_uniform_(self.w)
            #nn.init.xavier_uniform_(self.a_src_dst)
            nn.init.normal_(self.w_ori,std=1.0/math.sqrt(f_out))
            #nn.init.normal_(self.a_src_dst, mean=1.0, std=1.0)
            stdv = 1. / math.sqrt(self.a_src_dst.size(1)*self.a_src_dst.size(2))
            #nn.init.uniform_(self.a_src_dst, -stdv, stdv)
            nn.init.normal_(self.a_src_dst, mean=0, std=stdv)  


            
    def forward(self, h, inputr, A,ents2=None, relational=True):
        output = []
        N = h.size()
        for i in range(self.n_head):
            if i>=1:   #修
                if self.diag:
                    h = torch.mul(h, self.w[i-1])
                        
                    #inputr = torch.mul(inputr, self.w[i-1])
                else:
                    w= F.normalize(self.w_ori,dim=1)
                    w = torch.eye(w.size(-1)).cuda()-2*w.T.mm(w)
                    h = torch.mm(h,w)
            edge_h = h[A[2, :], :].mm(self.a_src_dst[i,0])
            edge_h = edge_h + inputr[A[1, :], :].mm(self.a_src_dst[i,1])
            #edge_hr = h[A[0, :], :].mm(self.a_src_dst[i,0])
            #edge_hr = edge_h - inputr[A[1, :], :].mm(self.a_src_dst[i,1])            
            edge_e = torch.exp(-self.leaky_relu(edge_h.squeeze())) # edge_e: 1 x 
            edge_e = F.dropout(edge_e, self.attn_dropout, training=self.training)   # edge_e: 1 x E            

            e_rowsum = torch.sparse.FloatTensor(A[[0],:].to(self.device), edge_e).to_dense().unsqueeze(1)
            edge_h = h[A[2, :], :] - inputr[A[1, :], :]
            if i ==self.n_head-1:
                del inputr
                del h
                gc.collect()
            edge_h = torch.mul(edge_h,edge_e.unsqueeze(0).transpose(0,1))        
            h_prime = torch.zeros(N).to(self.device)
            h_prime.index_add_(0,A[0, :].to(self.device),edge_h)         
            h_prime = h_prime.div(e_rowsum)
            output.append(h_prime.unsqueeze(0))
            del h_prime
            del edge_h
            del edge_e
            del e_rowsum
            gc.collect()
        del A
        gc.collect()   
        
        output = torch.cat(output, dim=0)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
    
    def __repr__(self):
        if self.diag:
            return self.__class__.__name__ + ' (' + str(self.f_out) + ' -> ' + str(self.f_out) + ') * ' + str(self.n_head) + ' heads'
        else:
            return self.__class__.__name__ + ' (' + str(self.f_in) + ' -> ' + str(self.f_out) + ') * ' + str(self.n_head) + ' heads'
        
class RMultiHeadGraphAttention2b(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    https://github.com/Diego999/pyGAT/blob/master/layers.py
    """
    def __init__(self, n_head, f_in, f_out, attn_dropout, diag=True, init=None, bias=False,Rmode=''):
        super(RMultiHeadGraphAttention2b, self).__init__()
        self.n_head = n_head
        self.f_in = f_in
        self.f_out = f_out
        self.diag = diag
        self.Rmode = Rmode
        
        if Rmode == 'mapping':
            #self.Map = Parameter(torch.Tensor(n_head-1, f_out, f_out))
            #nn.init.orthogonal(self.Map)
            self.Map_ori = Parameter(torch.Tensor(1,f_out))
   
        if self.diag:
            self.w = Parameter(torch.Tensor(n_head-1, 1, f_out)) #修
        else:
            self.w_ori = Parameter(torch.Tensor(1,f_out)) #修
            
        self.a_src_dst = Parameter(torch.Tensor(n_head, 2, f_out, 1))
            
        self.attn_dropout = attn_dropout
        self.leaky_relu = nn.LeakyReLU(negative_slope = 0.2)
        self.special_spmm = SpecialSpmm()
        if bias:
            self.bias = Parameter(torch.Tensor(f_out))
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter('bias', None)
        if init is not None and diag:
            init(self.w)
            stdv = 1. / math.sqrt(self.a_src_dst.size(1)*self.a_src_dst.size(2))
            #nn.init.uniform_(self.a_src_dst, -stdv, stdv)
            nn.init.normal_(self.a_src_dst, mean=0, std=stdv)
        else:
            #nn.init.xavier_uniform_(self.w)
            #nn.init.xavier_uniform_(self.a_src_dst)
            nn.init.normal_(self.w_ori,std=1.0/math.sqrt(f_out))
            #nn.init.normal_(self.a_src_dst, mean=1.0, std=1.0)
            stdv = 1. / math.sqrt(self.a_src_dst.size(1)*self.a_src_dst.size(2))
            #nn.init.uniform_(self.a_src_dst, -stdv, stdv)
            nn.init.normal_(self.a_src_dst, mean=0, std=stdv)  


            
    def forward(self, h, inputr, A,r_count=None,relational=True):
        output = []
        N = h.size()
        '''
        edge_r = torch.zeros((r_count.shape[0],N[1])).cuda()
        edge_h = h[A[2, :], :]- h[A[0, :], :]

        edge_r=edge_r.index_add(0,A[1, :].cuda(),edge_h)       
        edge_r=F.normalize(edge_r.div(r_count.unsqueeze(1))[0:3024]+0.00001)
        

        #with torch.no_grad():
        aij = F.relu(edge_r.mm(edge_r.T)-0.5)
        
        aij = aij.div(aij.sum(dim=1,keepdim=True))

        inputr= torch.cat((torch.mm(aij,r[0:3024]),r[3024].unsqueeze(0),-torch.mm(aij,r[0:3024])),dim=0)
        
        '''
        for i in range(self.n_head):
            
            if i>=1:   #修
                if self.diag:
                    h = torch.mul(h, self.w[i-1])                        
                    #inputr = torch.mul(inputr, self.w[i-1])
                else:
                    w= F.normalize(self.w_ori,dim=1)
                    w = torch.eye(w.size(-1)).cuda()-2*w.T.mm(w)
                    h = torch.mm(h,w)
                    

            edge_h = h[A[2, :], :].mm(self.a_src_dst[i,0])
            edge_h = edge_h + inputr[A[1, :], :].mm(self.a_src_dst[i,1])
            edge_e = torch.exp(-self.leaky_relu(edge_h.squeeze())) # edge_e: 1 x 
            edge_e = F.dropout(edge_e, self.attn_dropout, training=self.training)   # edge_e: 1 x E            
            #e_rowsum = self.special_spmm(A[[0,2],:], edge_e, torch.Size([N[0], N[0]]), torch.ones(size=(N[0], 1)).cuda()  if next(self.parameters()).is_cuda else torch.ones(size=(N[0], 1))) # e_rowsum: N x 1
            e_rowsum = torch.sparse.FloatTensor(A[[0],:].cuda(), edge_e).to_dense().unsqueeze(1)
            if self.Rmode=='abl':    
                edge_h = h[A[2, :], :]
            else:
                edge_h = h[A[2, :], :] - inputr[A[1, :], :]
            if i ==self.n_head-1:
                del h
                gc.collect()
            edge_h = torch.mul(edge_h,edge_e.unsqueeze(0).transpose(0,1))        
            h_prime = torch.zeros(N).cuda()
            h_prime.index_add_(0,A[0, :].cuda(),edge_h)         
            h_prime = h_prime.div(e_rowsum)
            output.append(h_prime.unsqueeze(0))
            del h_prime
            del edge_h
            del edge_e
            del e_rowsum
            gc.collect()
        del A
        gc.collect()   
        
        output = torch.cat(output, dim=0)
        if self.bias is not None:
            return output + self.bias,inputr
        else:
            return output,inputr
    
    def __repr__(self):
        if self.diag:
            return self.__class__.__name__ + ' (' + str(self.f_out) + ' -> ' + str(self.f_out) + ') * ' + str(self.n_head) + ' heads'
        else:
            return self.__class__.__name__ + ' (' + str(self.f_in) + ' -> ' + str(self.f_out) + ') * ' + str(self.n_head) + ' heads'


class RMultiHeadGraphAttention3(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    https://github.com/Diego999/pyGAT/blob/master/layers.py
    """
    def __init__(self, n_head, f_in, f_out, attn_dropout, diag=True, init=None, bias=False,Rmode='concat'):
        super(RMultiHeadGraphAttention3, self).__init__()
        self.n_head = n_head
        self.f_in = f_in
        self.f_out = f_out
        self.diag = diag
        self.Rmode = Rmode
        if self.diag:
            self.w = Parameter(torch.Tensor(n_head-1, 1, f_out)) #修
        else:
            self.w = Parameter(torch.Tensor(n_head-1, f_in, f_out)) #修
        
        self.rw_ori = Parameter(torch.Tensor( 1, f_out))
        nn.init.normal_(self.rw_ori,std=1.0/math.sqrt(f_out))
            
        self.a_src_dst = Parameter(torch.Tensor(n_head, 2, f_out, 1))
            
        self.attn_dropout = attn_dropout
        self.leaky_relu = nn.LeakyReLU(negative_slope = 0.2)
        #self.special_spmm = SpecialSpmm()
        if bias:
            self.bias = Parameter(torch.Tensor(f_out))
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter('bias', None)
        if init is not None and diag:
            init(self.w)
            stdv = 1. / math.sqrt(self.a_src_dst.size(1)*self.a_src_dst.size(2))
            nn.init.uniform_(self.a_src_dst, -stdv, stdv)
        else:
            nn.init.xavier_uniform_(self.w)
            nn.init.xavier_uniform_(self.a_src_dst)

    def forward(self, h,A,r_count, relational=True):
        output = []
        N = h.size()
        inputr = torch.zeros((r_count.shape[0],N[1])).cuda()
        edge_h = (h[A[2, :], :]- h[A[0, :], :]).div(r_count.unsqueeze(1))
        inputr.index_add_(0,A[1, :].cuda(),edge_h)
        edge_r = inputr.div(r_count.unsqueeze(1))
        #with torch.no_grad():
        aij = edge_r.mm(edge_r.T)
        aij = aij.div(aij.sum(dim=1))
        inputr = torch.mm(aij,inputr)
        
            
            
            
        #Map= F.normalize(self.rw_ori,dim=1)
        #Map = torch.eye(Map.size(-1)).cuda()-2*Map.T.mm(Map)
        #inputr = inputr.mm(Map)
        for i in range(self.n_head):
            if i>=1:   #修
                if self.diag:
                    h = torch.mul(h, self.w[i-1])
                    #inputr = torch.mul(inputr, self.w[i-1])
                else:
                    h = torch.mm(h, self.w[i-1])
            
            edge_h = h[A[2, :], :].mm(self.a_src_dst[i,0])
            edge_h = edge_h + inputr[A[1, :], :].mm(self.a_src_dst[i,1])
            edge_e = torch.exp(-self.leaky_relu(edge_h.squeeze())) # edge_e: 1 x 
            edge_e = F.dropout(edge_e, self.attn_dropout, training=self.training)   # edge_e: 1 x E
            #print(edge_e)

            
            e_rowsum = torch.sparse.FloatTensor(A[[0],:].cuda(), edge_e).to_dense().unsqueeze(1)
            
            edge_h = h[A[2, :], :] - inputr[A[1, :], :]
            
            if i ==self.n_head-1:
                del inputr      
                del h
                gc.collect()
            edge_h = torch.mul(edge_h,edge_e.unsqueeze(0).transpose(0,1))        
            h_prime = torch.zeros(N).cuda()
            h_prime.index_add_(0,A[0, :].cuda(),edge_h)
            
            h_prime = h_prime.div(e_rowsum)
            
            output.append(h_prime.unsqueeze(0))
            del h_prime
            del edge_h
            del edge_e
            del e_rowsum
            gc.collect()
        del A        
        output = torch.cat(output, dim=0)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
     
    def __repr__(self):
        if self.diag:
            return self.__class__.__name__ + ' (' + str(self.f_out) + ' -> ' + str(self.f_out) + ') * ' + str(self.n_head) + ' heads'
        else:
            return self.__class__.__name__ + ' (' + str(self.f_in) + ' -> ' + str(self.f_out) + ') * ' + str(self.n_head) + ' heads'

class LPlayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    https://github.com/Diego999/pyGAT/blob/master/layers.py
    """
    def __init__(self, n_head, f_in, f_out, attn_dropout, diag=True, init=None, bias=False,Rmode=''):
        super(LPlayer, self).__init__()
        self.n_head = n_head
        self.f_in = f_in
        self.f_out = f_out
        self.diag = diag
        self.Rmode = Rmode
        
        if Rmode == 'mapping':
            #self.Map = Parameter(torch.Tensor(n_head-1, f_out, f_out))
            #nn.init.orthogonal(self.Map)
            self.Map_ori = Parameter(torch.Tensor(1,f_out))
   
        if self.diag:
            self.w = Parameter(torch.Tensor(n_head-1, 1, f_out)) #修
        else:
            self.w_ori = Parameter(torch.Tensor(1,f_out)) #修
            
        self.a_src_dst = Parameter(torch.Tensor(n_head, 2, f_out, 1))
            
        self.attn_dropout = attn_dropout
        self.leaky_relu = nn.LeakyReLU(negative_slope = 0.2)
        
        if bias:
            self.bias = Parameter(torch.Tensor(f_out))
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter('bias', None)
        if init is not None and diag:
            init(self.w)
            stdv = 1. / math.sqrt(self.a_src_dst.size(1)*self.a_src_dst.size(2))
            #nn.init.uniform_(self.a_src_dst, -stdv, stdv)
            nn.init.normal_(self.a_src_dst, mean=0, std=stdv)
        else:
            #nn.init.xavier_uniform_(self.w)
            #nn.init.xavier_uniform_(self.a_src_dst)
            nn.init.normal_(self.w_ori,std=1.0/math.sqrt(f_out))
            #nn.init.normal_(self.a_src_dst, mean=1.0, std=1.0)
            stdv = 1. / math.sqrt(self.a_src_dst.size(1)*self.a_src_dst.size(2))
            #nn.init.uniform_(self.a_src_dst, -stdv, stdv)
            nn.init.normal_(self.a_src_dst, mean=0, std=stdv)  


            
    def forward(self, h, inputr, A,cfdc,sup_ents,trainMode=True):
        output = []
        N = h.size()
        for i in range(self.n_head):
            if i>=1:   #修
                if self.diag:
                    h = torch.mul(h, self.w[i-1])
                        
                    #inputr = torch.mul(inputr, self.w[i-1])
                else:
                    w= F.normalize(self.w_ori,dim=1)
                    w = torch.eye(w.size(-1)).cuda()-2*w.T.mm(w)
                    h = torch.mm(h,w)

            #edge_h = torch.norm(h[A[0, :], :]+ inputr[A[1, :], :]-h[A[2, :], :],dim=1)
            #edge_e2 = torch.exp(-self.leaky_relu(edge_h.squeeze().mul(cfdc))) # edge_e: 1 x 

            #edge_e1 = torch.exp(cfdc)
            #edge_e1 = torch.ones(cfdc.size()).cuda()
            edge_h = h[A[2, :], :].mm(self.a_src_dst[i,0])
            edge_h = edge_h + inputr[A[1, :], :].mm(self.a_src_dst[i,1])
            edge_e1 = torch.exp(-self.leaky_relu(edge_h.squeeze()))# edge_e: 1 x 
            #edge_e = F.dropout(edge_e, self.attn_dropout, training=self.training)   # edge_e: 1 x E            

            e_rowsum1 = torch.sparse.FloatTensor(A[[0],:].cuda(), edge_e1).to_dense().unsqueeze(1)
            #e_rowsum2 = torch.sparse.FloatTensor(A[[0],:].cuda(), edge_e2).to_dense().unsqueeze(1)
            edge_h = h[A[2, :], :] - inputr[A[1, :], :]
            if i ==self.n_head-1:
                del inputr
                del h
                gc.collect()
            edge_h1= torch.mul(edge_h,edge_e1.unsqueeze(0).transpose(0,1))        
            h_prime1 = torch.zeros(N).cuda()
            h_prime1.index_add_(0,A[0, :].cuda(),edge_h1)         
            h_prime1 = h_prime1.div(e_rowsum1)
            
            #edge_h2= torch.mul(edge_h,edge_e2.unsqueeze(0).transpose(0,1))        
            #h_prime2 = torch.zeros(N).cuda()
            #h_prime2.index_add_(0,A[0, :].cuda(),edge_h2)         
            #h_prime1[sup_ents] = h_prime2.div(e_rowsum2)[sup_ents]            
            output.append(h_prime1.unsqueeze(0))
            del h_prime1
            del edge_h1
            del edge_e1
            del e_rowsum1
            gc.collect()
        del A
        gc.collect()   
        
        output = torch.cat(output, dim=0)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
    
    def __repr__(self):
        if self.diag:
            return self.__class__.__name__ + ' (' + str(self.f_out) + ' -> ' + str(self.f_out) + ') * ' + str(self.n_head) + ' heads'
        else:
            return self.__class__.__name__ + ' (' + str(self.f_in) + ' -> ' + str(self.f_out) + ') * ' + str(self.n_head) + ' heads'


        
class LPlayer3(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    https://github.com/Diego999/pyGAT/blob/master/layers.py
    """
    def __init__(self, n_head, f_in, f_out, attn_dropout, diag=True, init=None, bias=False,Rmode=''):
        super(LPlayer, self).__init__()
        self.n_head = n_head
        self.f_in = f_in
        self.f_out = f_out
        self.diag = diag
        self.Rmode = Rmode
        
        if Rmode == 'mapping':
            #self.Map = Parameter(torch.Tensor(n_head-1, f_out, f_out))
            #nn.init.orthogonal(self.Map)
            self.Map_ori = Parameter(torch.Tensor(1,f_out))
   
        if self.diag:
            self.w = Parameter(torch.Tensor(n_head-1, 1, f_out)) #修
        else:
            self.w_ori = Parameter(torch.Tensor(1,f_out)) #修
            
        self.a_src_dst = Parameter(torch.Tensor(n_head, 2, f_out, 1))
            
        self.attn_dropout = attn_dropout
        self.leaky_relu = nn.LeakyReLU(negative_slope = 0.2)
        self.special_spmm = SpecialSpmm()
        if bias:
            self.bias = Parameter(torch.Tensor(f_out))
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter('bias', None)
        if init is not None and diag:
            init(self.w)
            stdv = 1. / math.sqrt(self.a_src_dst.size(1)*self.a_src_dst.size(2))
            #nn.init.uniform_(self.a_src_dst, -stdv, stdv)
            nn.init.normal_(self.a_src_dst, mean=0, std=stdv)
        else:
            #nn.init.xavier_uniform_(self.w)
            #nn.init.xavier_uniform_(self.a_src_dst)
            nn.init.normal_(self.w_ori,std=1.0/math.sqrt(f_out))
            #nn.init.normal_(self.a_src_dst, mean=1.0, std=1.0)
            stdv = 1. / math.sqrt(self.a_src_dst.size(1)*self.a_src_dst.size(2))
            #nn.init.uniform_(self.a_src_dst, -stdv, stdv)
            nn.init.normal_(self.a_src_dst, mean=0, std=stdv)  


            
    def forward(self, h, inputr, A,cfdc,sup_ents,trainMode=True):
        output = []
        N = h.size()
        for i in range(self.n_head):
            if i>=2:   #修
                if self.diag:
                    h = torch.mul(h, self.w[i-1])
                        
                    #inputr = torch.mul(inputr, self.w[i-1])
                else:
                    w= F.normalize(self.w_ori,dim=1)
                    w = torch.eye(w.size(-1)).cuda()-2*w.T.mm(w)
                    h = torch.mm(h,w)

            edge_h = torch.norm(h[A[0, :], :]+ inputr[A[1, :], :]-h[A[2, :], :],dim=1)
            #edge_e2 = torch.exp(-self.leaky_relu(edge_h.squeeze().mul(cfdc))) # edge_e: 1 x 

            #edge_e = torch.exp(cfdc)

            #edge_h = h[A[2, :], :].mm(self.a_src_dst[i,0])
            #edge_h = edge_h + inputr[A[1, :], :].mm(self.a_src_dst[i,1])
            edge_e1 = torch.exp(-self.leaky_relu(edge_h.squeeze().mul(cfdc))) # edge_e: 1 x 
            #edge_e = F.dropout(edge_e, self.attn_dropout, training=self.training)   # edge_e: 1 x E            
            #e_rowsum = self.special_spmm(A[[0,2],:], edge_e, torch.Size([N[0], N[0]]), torch.ones(size=(N[0], 1)).cuda()  if next(self.parameters()).is_cuda else torch.ones(size=(N[0], 1))) # e_rowsum: N x 1
            e_rowsum1 = torch.sparse.FloatTensor(A[[0],:].cuda(), edge_e1).to_dense().unsqueeze(1)
            #e_rowsum2 = torch.sparse.FloatTensor(A[[0],:].cuda(), edge_e2).to_dense().unsqueeze(1)
            edge_h = h[A[2, :], :] - inputr[A[1, :], :]
            if i ==self.n_head-1:
                del inputr
                del h
                gc.collect()
            edge_h1= torch.mul(edge_h,edge_e1.unsqueeze(0).transpose(0,1))        
            h_prime1 = torch.zeros(N).cuda()
            h_prime1.index_add_(0,A[0, :].cuda(),edge_h1)         
            h_prime1 = h_prime1.div(e_rowsum1)
            
            #edge_h2= torch.mul(edge_h,edge_e2.unsqueeze(0).transpose(0,1))        
            #h_prime2 = torch.zeros(N).cuda()
            #h_prime2.index_add_(0,A[0, :].cuda(),edge_h2)         
            #h_prime1[sup_ents] = h_prime2.div(e_rowsum2)[sup_ents]            
            output.append(h_prime1.unsqueeze(0))
            del h_prime1
            del edge_h
            del edge_e1
            del e_rowsum1
            gc.collect()
        del A
        gc.collect()   
        
        output = torch.cat(output, dim=0)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
    
    def __repr__(self):
        if self.diag:
            return self.__class__.__name__ + ' (' + str(self.f_out) + ' -> ' + str(self.f_out) + ') * ' + str(self.n_head) + ' heads'
        else:
            return self.__class__.__name__ + ' (' + str(self.f_in) + ' -> ' + str(self.f_out) + ') * ' + str(self.n_head) + ' heads'


class Highway(nn.Module):
    def __init__(self,d):
        super(Highway, self).__init__()
        self.w = Parameter(torch.Tensor(d,d))
        self.bias = Parameter(torch.Tensor(d))
        nn.init.xavier_uniform_(self.w)
        nn.init.constant_(self.bias, 0)
        
             
        
    def forward(self,layer1, layer2):
        t1 = torch.sigmoid(torch.mm(layer1, self.w)+self.bias)
        t2 = 1.0 -t1
        return t1 * layer1 + t2 * layer2
        
        