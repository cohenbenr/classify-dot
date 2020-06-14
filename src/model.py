import numpy as np
import torch
import torch.nn.functional as F

from torch import nn

class MarginRankingLoss(nn.Module):
    def __init__(self, margin=1., aggregate=torch.mean):
        super(MarginRankingLoss, self).__init__()
        self.margin = margin
        self.aggregate = aggregate

    def forward(self, positive_similarity, negative_similarity):
        return self.aggregate(
            torch.clamp(self.margin - positive_similarity + negative_similarity, min=0))
    

class InnerProductSimilarity(nn.Module):
    def __init__(self):
        super(InnerProductSimilarity, self).__init__()

    def forward(self, a, b):
        # a => B x [n_a x] dim, b => B x [n_b x] dim

        if a.dim() == 2:
            a = a.unsqueeze(1)  # B x n_a x dim

        if b.dim() == 2:
            b = b.unsqueeze(1)  # B x n_b x dim

        return torch.bmm(a, b.transpose(2, 1))  # B x n_a x n_b


class StarSpace(nn.Module):
    def __init__(self, d_embed, vocabulary, input_embedding = None, k_neg = 3, max_norm=20):
        super(StarSpace, self).__init__()

        self.n_input = len(vocabulary)
        self.vocab = vocabulary
        self.k_neg = k_neg
        
        if input_embedding is None:
            self.embeddings = nn.Embedding(self.n_input, d_embed, max_norm=max_norm)
        else:
            self.embeddings = input_embedding
            
    def embed_doc(self,d,normalize=False):
        positions = []
        for t in d:
            try:
                positions.append(self.vocab[t])
            except KeyError:
                pass
        output = torch.sum(self.embeddings(torch.LongTensor(positions)),dim=0)
        output[output != output] = 0 #necessary for documents with all unseen vocabs
        
        if normalize:
            output = output / output.norm()
        return output
    
    def forward(self, docs):       
        l_batch = []
        r_batch = []
        neg_batch = []
        
        for i in range(len(docs)):
            #Positive similarity
            s = docs[i].split('\t') #sentences
            if type(s) == str: #only one sentence in s
                a = s
                b = s
            else:
                a, b = np.random.choice(s, 2, False)

            a = a.split()
            b = b.split()

            a_emb = self.embed_doc(a)
            b_emb = self.embed_doc(b)

            l_batch.append(a_emb)
            r_batch.append(b_emb)

            #Negative similarity
            negs = []
            while len(negs) < self.k_neg:
                index = np.random.choice(len(docs))
                
                if index != i: #if it's not from the same document
                    c = docs[index].split('\t')
                    c = np.random.choice(c, 1)[0].split()
                    #c_emb = self.embed_doc(c, normalize=True)
                    c_emb = self.embed_doc(c)
                    negs.append(c_emb)

            neg_batch.append(torch.stack(negs))
        
        l_batch = torch.stack(l_batch)
        r_batch = torch.stack(r_batch)
        neg_batch = torch.stack(neg_batch)

        l_batch = l_batch.unsqueeze(1)
        r_batch = r_batch.unsqueeze(1)
        
        return l_batch, r_batch, neg_batch
    

# class StarSpace_old(nn.Module):
#     def __init__(self, d_embed, n_input, n_output, similarity, max_norm=10, aggregate=torch.sum):
#         super(StarSpace, self).__init__()

#         self.n_input = n_input
#         self.n_output = n_output
#         self.similarity = similarity
#         self.aggregate = aggregate
        
#         self.input_embedding = nn.Embedding(n_input, d_embed, max_norm=max_norm)
#         self.output_embedding = nn.Embedding(n_output, d_embed, max_norm=max_norm)

#     def forward(self, input=None, output=None):
#         input_repr, output_repr = None, None
        
#         if input is not None:
            
#             if input.dim() == 1:
#                 input = input.unsqueeze(-1)

#             input_emb = self.input_embedding(input)  # B x L_i x dim
#             input_repr = self.aggregate(input_emb, dim=1)  # B x dim
        
#         if output is not None:
#             if output.dim() == 1:
#                 output = output.unsqueeze(-1)  # B x L_o

#             output_emb = self.output_embedding(output)  # B x L_o x dim
#             output_repr = self.aggregate(output_emb, dim=1)  # B x dim
        
#         return input_repr, output_repr