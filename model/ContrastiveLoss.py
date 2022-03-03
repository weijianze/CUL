"""
Author: BigLiu (jianze.wei@cripac.ia.ac.cn)
Date: January 20, 2021
Refer from:
link-1: https://github.com/HobbitLong/SupContrast/blob/master/losses.py
link-2: https://github.com/Spijkervet/SimCLR/blob/654f05f107ce17c0a9db385f298a2dc6f8b3b870/modules/nt_xent.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
# import pdb

# import argparse

class ContrastiveLoss(nn.Module):
    def __init__(self,args):
        super(ContrastiveLoss, self).__init__()
        self.similarity_f = nn.CosineSimilarity(dim=2)
        self.args     = args

    def forward(self, embedding, labels=None, predy=None, mu=None, logvar=None):
        batch_size = embedding[0].size(0)
        embedding_num = len(embedding)
        loss = {}
        device = (torch.device('cuda')
            if embedding[0].is_cuda
            else torch.device('cpu'))

        if labels is None:            
            mask = torch.eye(batch_size, dtype=bool).to(device)
            mask_positive = mask.repeat(embedding_num, embedding_num)
        else:
            if type(labels)==tuple:
                labels = torch.cat(labels, dim=0)
            labels = labels.contiguous().view(-1, 1)
            mask_positive = (torch.eq(labels, labels.T)>0).to(device)
            labels = labels.squeeze()
        embedding = torch.cat(embedding,dim=0)

        if not embedding.size(0)%embedding_num == 0:
            raise ValueError('There are some errors happened in `embedding` from different views')
                     
        mask_sampling = torch.ones_like(mask_positive,dtype=bool) ^ torch.eye(mask_positive.size(0),dtype=bool).to(device)   
        sim = self.similarity_f(embedding.unsqueeze(1), embedding.unsqueeze(0)) / self.args.temperature
        
        mask_positive = mask_positive[mask_sampling].reshape(mask_positive.size(0),-1)
        sim = sim[mask_sampling].reshape(sim.size(0),-1)
        logits = F.log_softmax(sim,dim=1)
        loss['contrastive'] = -logits[mask_positive].mean()


        # Arcface loss 
        if predy is not None:
            if self.args.loss_mode == 'focal_loss':
                logp = F.cross_entropy(predy, labels, reduce=False)
                prob = torch.exp(-logp)
                # loss.append(((1-prob) ** self.args.loss_power * logp).mean())
                loss['classification'] = ((1-prob) ** self.args.loss_power * logp).mean()
            elif self.args.loss_mode == 'hardmining':
                batchsize = predy.shape[0]
                logp      = F.cross_entropy(predy, labels, reduce=False)
                ind_sorted = torch.argsort(-logp) # from big to small
                num_hard  = int(self.args.hard_ratio * self.args.batch_size)
                hard_idx  = ind_sorted[:num_hard]
                # loss.append(torch.sum(F.cross_entropy(predy[hard_idx], labels[hard_idx])))
                loss['classification'] = torch.sum(F.cross_entropy(predy[hard_idx], labels[hard_idx]))
            else: # navie-softmax
                # loss.append(F.cross_entropy(predy, labels))
                loss['classification'] = F.cross_entropy(predy, labels)
        else:
            loss['classification'] = None


        if (mu is not None) and (logvar is not None):
            kl_loss = -(1 + logvar - mu.pow(2) - logvar.exp()) / 2
            # loss.append(kl_loss.sum(dim=1).mean())
            loss['kl'] = kl_loss.sum(dim=1).mean()
        else:
            loss['kl'] = None
            
        return loss


# if __name__ == '__main__':

#     parser = argparse.ArgumentParser(description='PyTorch for DUL-classification')
#     parser.add_argument('--loss_mode',  type=str,    default='hardmining',       choices=['ce', 'focal_loss', 'hardmining'])
#     parser.add_argument('--hard_ratio', type=float,  default=0.9)          # hardmining
#     parser.add_argument('--loss_power', type=int,    default=2)            # focal_loss
#     parser.add_argument('--temperature',   type=float,   default=0.07)    # from NIPS2020: supervised contrastive learning
#     parser.add_argument('--batch_size',  type=int,   default=3)      # TODO | 300
#     args = parser.parse_args()
#     criterion_self = ContrastiveArcFaceLoss(args)

#     for i in range(1):
#         data_1 = torch.randn(3,128)
#         data_2 = torch.randn(3,128)
#         predy_1 = torch.randn(3,3)
#         predy_2 = torch.randn(3,3)
#         mu = torch.randn(3,128)
#         logvar = torch.randn(3,128)
#         labels = torch.empty(3, dtype=torch.long).random_(3)

#         criterion_self = criterion_self.cuda()
#         labels = labels.cuda()
#         data_1 = data_1.cuda()
#         data_2 = data_2.cuda()
#         predy_1 = predy_1.cuda()
#         predy_2 = predy_2.cuda()
#         logvar = logvar.cuda()
#         mu = mu.cuda()

#         loss_1 = criterion_self(torch.cat((data_1,data_2),dim=0))
#         loss_2 = criterion_self(torch.cat((data_1,data_2),dim=0),torch.cat((labels,labels),dim=0))
#         loss_3 = criterion_self(torch.cat((data_1,data_2),dim=0),torch.cat((labels,labels),dim=0),torch.cat((predy_1,predy_2),dim=0))
#         loss_4 = criterion_self(torch.cat((data_1,data_2),dim=0),torch.cat((labels,labels),dim=0),torch.cat((predy_1,predy_2),dim=0),mu,logvar)
#         pdb.set_trace()
#         print('Self: {:.4f}  {:.4f}  {:.4f}  {:.4f}  '.format(loss_1, loss_2, loss_3, loss_4))