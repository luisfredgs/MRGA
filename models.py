#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  File name:    models.py
  Author:       locke
  Date created: 2018/10/5 下午2:37
"""

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *
import gc

class GAT(nn.Module):
    def __init__(self, n_units, n_heads, dropout, attn_dropout, instance_normalization, diag):
        super(GAT, self).__init__()
        self.num_layer = len(n_units) - 1
        self.dropout = dropout
        self.inst_norm = instance_normalization
        if self.inst_norm:
            self.norm = nn.InstanceNorm1d(n_units[0], momentum=0.0, affine=True)
        self.layer_stack = nn.ModuleList()
        self.diag = diag
        for i in range(self.num_layer):
            f_in = n_units[i] * n_heads[i - 1] if i else n_units[i]
            self.layer_stack.append(MultiHeadGraphAttention(n_heads[i], f_in, n_units[i + 1], attn_dropout, diag, nn.init.ones_, False))

    def forward(self, x, adj):
        if self.inst_norm:
            x = self.norm(x)
        for i, gat_layer in enumerate(self.layer_stack):
            if i + 1 < self.num_layer:
                x = F.dropout(x, self.dropout, training=self.training)
            x = gat_layer(x, adj)
            if self.diag:
                x = x.mean(dim=0)
            if i + 1 < self.num_layer:
                if self.diag:
                    x = F.elu(x)
                else:
                    x = F.elu(x.transpose(0, 1).contiguous().view(adj.size(0), -1))
        if not self.diag:
            x = x.mean(dim=0)

        return x

class RGAT(nn.Module):
    def __init__(self, n_units, n_heads, dropout, attn_dropout, instance_normalization,l2norm, diag,Rmode,RGATmode):
        super(RGAT, self).__init__()
        self.num_layer = len(n_units) - 1
        self.dropout = dropout
        self.inst_norm = instance_normalization
        self.l2norm = l2norm
        self.Rmapping = Parameter(torch.Tensor(1,n_units[0]))
        self.Rmode = Rmode
        if self.inst_norm:
            self.norm = nn.InstanceNorm1d(n_units[0], momentum=0.0, affine=True)
        self.layer_stack = nn.ModuleList()
        self.diag = diag
        self.RGATmode = RGATmode
        
        self.highway= Highway(n_units[0])
        for i in range(self.num_layer):
            f_in = n_units[i] * n_heads[i - 1] if i else n_units[i]
            if RGATmode==2:
                self.layer_stack.append(RMultiHeadGraphAttention2(n_heads[i], f_in, n_units[i + 1], attn_dropout, diag, nn.init.ones_, False,Rmode))
            elif RGATmode==4:
                self.layer_stack.append(RMultiHeadGraphAttention2b(n_heads[i], f_in, n_units[i + 1], attn_dropout, diag, nn.init.ones_, False,Rmode))
            else:
                self.layer_stack.append(RMultiHeadGraphAttentionab(n_heads[i], f_in, n_units[i + 1], attn_dropout, diag, nn.init.ones_, False,Rmode))
    def forward(self,x,r,A,r_count=None,ents2=None,rels2=None):
        highway_flag=False
        if self.l2norm:
            r = F.normalize(r)  
        if self.Rmode == 'rmapping':
            Map= F.normalize(self.Rmapping,dim=1)
            Map = torch.eye(Map.size(-1)).cuda()-2*Map.T.mm(Map)
            r[rels2] =r[rels2].mm(Map)
            
        if self.inst_norm:
            x = self.norm(x)
        if self.l2norm:
            x = F.normalize(x)  
            
        if highway_flag==True:
            
            x = F.dropout(x, self.dropout, training=self.training)
            x0 = self.layer_stack[0](x, r, A)
            x0 = x0.mean(dim=0)
            x0 = F.elu(x0)
            
            x = self.layer_stack[1](x0, r, A)
            x = x.mean(dim=0)
            x = self.highway(x0,x)
            
        else:
            if self.RGATmode == 4:

             
                x = F.dropout(x, self.dropout, training=self.training)

                x,r = self.layer_stack[0](x, r, A)
                x = x.mean(dim=0)

                edge_r = torch.zeros((r_count.shape[0],x.size()[1]*2)).cuda()
                edge_h = torch.cat((x[A[2, :], :],x[A[0, :], :]),dim=1)

                edge_r=edge_r.index_add(0,A[1, :].cuda(),edge_h)       
                edge_r=F.normalize(edge_r.div(r_count.unsqueeze(1))+0.00001)


                #with torch.no_grad():
                aij = F.relu(edge_r.mm(edge_r.T)-0.1)

                aij = aij.div(aij.sum(dim=1,keepdim=True))

                r= torch.mm(aij,r)
                if self.l2norm:
                    r = F.normalize(r)
                
                x0 = F.elu(x)
                x,r= self.layer_stack[1](x0, r, A)
                x = x.mean(dim=0)
                
            else:
                for i, gat_layer in enumerate(self.layer_stack):        

                    if i + 1 < self.num_layer:  #2
                        x = F.dropout(x, self.dropout, training=self.training)

                    if self.RGATmode == 4:
                        x,r = gat_layer(x, r, A)
                        x = x.mean(dim=0)
                        if i==0:
                            edge_r = torch.zeros((r_count.shape[0],x.size()[1]*2)).cuda()
                            edge_h = torch.cat((x[A[2, :], :],x[A[0, :], :]),dim=1)

                            edge_r=edge_r.index_add(0,A[1, :].cuda(),edge_h)       
                            edge_r=F.normalize(edge_r.div(r_count.unsqueeze(1))+0.00001)


                            #with torch.no_grad():
                            aij = F.relu(edge_r.mm(edge_r.T)-0.3)

                            aij = aij.div(aij.sum(dim=1,keepdim=True))

                            r= torch.mm(aij,r)
                            if self.l2norm:
                                r = F.normalize(r)
                    else:
                        #del r_count
                        #gc.collect()
                        x = gat_layer(x, r, A)
                        #x = self.layer_stack[0](x, r, A,ents2)

                    #if self.diag:
                        x = x.mean(dim=0)

                    if i + 1 < self.num_layer:
                        x = F.elu(x)
                        #if self.diag:
                            #x = F.elu(x)
                        #else:
                            #x = F.elu(x.transpose(0, 1).contiguous().view(adj.size(0), -1))
                #if not self.diag:
                    #x = x.mean(dim=0)
                    if i==0:
                        x0=x
        #if not self.diag:
            #x = x.mean(dim=0)
        del A
        del r
        gc.collect()

        return torch.cat((x0,x),dim=1)
        #return 0.5*(x0+x)
    

class RGATM(nn.Module):
    def __init__(self, n_units, n_heads, dropout, attn_dropout, instance_normalization,l2norm, diag,Rmode,RGATmode):
        super(RGATM, self).__init__()
        self.num_layer = len(n_units) - 1
        self.dropout = dropout
        self.inst_norm = instance_normalization
        self.l2norm = l2norm
        self.Rmapping = Parameter(torch.Tensor(1,n_units[0]))
        self.Rmode = Rmode
        if self.inst_norm:
            self.norm = nn.InstanceNorm1d(n_units[0], momentum=0.0, affine=True)
        self.layer_stack = nn.ModuleList()
        self.diag = diag
        self.RGATmode = RGATmode
        f_in = n_units[0] * n_heads[0]

        self.layer_stack.append(RMultiHeadGraphAttention2m(n_heads[0], f_in, n_units[0], attn_dropout, diag, nn.init.ones_, False,Rmode,device='cuda:0').to('cuda:0'))
        self.layer_stack.append(RMultiHeadGraphAttention2m(n_heads[0], f_in, n_units[0], attn_dropout, diag, nn.init.ones_, False,Rmode,device='cuda:1').to('cuda:1'))
    def forward(self,x,r,A,r_count=None,ents2=None,rels2=None):
        if self.l2norm:
            r = F.normalize(r)  

        if self.inst_norm:
            x = self.norm(x)
        if self.l2norm:
            x = F.normalize(x)  
        x = self.layer_stack[0](x, r, A)
        x = x.mean(dim=0)
        x = F.elu(x)
        x = self.layer_stack[1](x.to('cuda:1'), r.to('cuda:1'), A)
        x = x.mean(dim=0)

        del A
        del r
        gc.collect()
        return x
        #return torch.cat((x0,x),dim=1)
        #return 0.5*(x0+x)
class RGAT3(nn.Module):
    def __init__(self, n_units, n_heads, dropout, attn_dropout, instance_normalization,l2norm, diag,Rmode,RGATmode=3):
        super(RGAT3, self).__init__()
        self.num_layer = len(n_units) - 1
        self.dropout = dropout
        self.inst_norm = instance_normalization
        self.l2norm = l2norm
        if self.inst_norm:
            self.norm = nn.InstanceNorm1d(n_units[0], momentum=0.0, affine=True)
        self.layer_stack = nn.ModuleList()
        self.diag = diag
        for i in range(self.num_layer):
            f_in = n_units[i] * n_heads[i - 1] if i else n_units[i]
            self.layer_stack.append(RMultiHeadGraphAttention3(n_heads[i], f_in, n_units[i + 1], attn_dropout, diag, nn.init.ones_, False,Rmode))

    def forward(self,x,A,r_count):
        if self.inst_norm:
            x = self.norm(x)
        if self.l2norm:
            x = F.normalize(x)  
        for i, gat_layer in enumerate(self.layer_stack):        
          
            if i + 1 < self.num_layer:
                x = F.dropout(x, self.dropout, training=self.training)

            x = gat_layer(x,A,r_count)
            if self.diag:
                x = x.mean(dim=0)
            if i + 1 < self.num_layer:
                if self.diag:
                    x = F.elu(x)
                else:
                    x = F.elu(x.transpose(0, 1).contiguous().view(adj.size(0), -1))
        if not self.diag:
            x = x.mean(dim=0)

        return x
        
class LPmodel(nn.Module):
    def __init__(self, n_units, n_heads, dropout, attn_dropout, instance_normalization,l2norm, diag,Rmode,RGATmode):
        super(LPmodel, self).__init__()
        self.num_layer = len(n_units) - 1
        self.dropout = dropout
        self.inst_norm = instance_normalization 
        self.l2norm = l2norm
        self.Rmapping = Parameter(torch.Tensor(1,n_units[0]))
        self.Rmode = Rmode
        if self.inst_norm:
            self.norm = nn.InstanceNorm1d(n_units[0], momentum=0.0, affine=True)
        self.layer_stack = nn.ModuleList()
        self.diag = diag
        self.RGATmode = RGATmode
        for i in range(self.num_layer):
            f_in = n_units[i] * n_heads[i - 1] if i else n_units[i]
            if RGATmode==2:
                self.layer_stack.append(LPlayer(n_heads[i], f_in, n_units[i + 1], attn_dropout, diag, nn.init.ones_, False,Rmode))
            elif RGATmode==4:
                self.layer_stack.append(RMultiHeadGraphAttention4(n_heads[i], f_in, n_units[i + 1], attn_dropout, diag, nn.init.ones_, False,Rmode))
            else:
                self.layer_stack.append(RMultiHeadGraphAttention(n_heads[i], f_in, n_units[i + 1], attn_dropout, diag, nn.init.ones_, False,Rmode))
    def forward(self,x,r,A,cfdc,sup_ents,trainMode):
        if self.l2norm:
            r = F.normalize(r)  
            
        if self.inst_norm:
            x = self.norm(x)
        if self.l2norm:
            x = F.normalize(x)  
        for i, gat_layer in enumerate(self.layer_stack):        
          
            if i + 1 < self.num_layer:
                x = F.dropout(x, self.dropout, training=self.training)

            x = gat_layer(x, r, A,cfdc,sup_ents,trainMode)
                
            #if self.diag:
            x = x.mean(dim=0)
            if i + 1 < self.num_layer:
                x = F.elu(x)
                #if self.diag:
                    #x = F.elu(x)
                #else:
                    #x = F.elu(x.transpose(0, 1).contiguous().view(adj.size(0), -1))
            if i==0:
                x0=x
        #if not self.diag:
            #x = x.mean(dim=0)
        del A
        del r
        gc.collect()

        return torch.cat((x0,x),dim=1)
        
    
    
    
    