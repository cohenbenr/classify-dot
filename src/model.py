import numpy as np
import torch
import torch.nn.functional as F

from torch import nn

class StarSpace(nn.Module):
    def __init__(self, d_embed, vocabulary, input_embedding = None, k_neg = 3, max_norm=20):
        super(StarSpace, self).__init__()

        self.n_input = len(vocabulary)
        self.vocab = vocabulary
        self.k_neg = k_neg
        self.d_embed = d_embed
        
        if input_embedding is None:
            self.embeddings = nn.Embedding(self.n_input, self.d_embed, max_norm=max_norm)
        else:
            self.embeddings = input_embedding
        
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
            
        self.embeddings.to(self.device)
        
    def get_positions(self,train):
        """ Get the positions of every word.  Return list of list of tensors """
        train_pos = []
        for i,doc in enumerate(train): 
            # For each document
            sentences = doc.split('\t')
            doc_positions = []
            for s in sentences: 
                # For each sentence
                positions = []
                s = s.split()
                
                for tok in s: 
                    # For each word
                    try:
                        positions.append(self.vocab[tok])
                    except KeyError:
                        pass
                
                #TEMPORARY FIX- a totally neutral but not too common word
                if len(positions) < 1:
                    positions.append(self.vocab['able'])
                    
                doc_positions.append(torch.LongTensor(positions))

            train_pos.append(doc_positions)
        
        return np.array(train_pos)
    
    def embed_doc(self,d,normalize=False):
        """ Takes a tensor of positions and embeds it """
        output = torch.sum(self.embeddings(d),dim=0)
        #output[output != output] = 0 #necessary for documents with all unseen vocabs
        
        output.to(self.device)
        
        if normalize:
            output = output / output.norm()
        return output
    
    def forward(self, docs):       
        l_batch = []
        r_batch = []
        neg_batch = []
        
        for i,s in enumerate(docs):
            #Positive similarity between sentences
            if (type(s) == str) or (len(s) <= 1): #only one sentence in s
                a = s[0]
                b = s[0]
            else:
                choices = np.random.choice(len(s), 2, False)
                a = s[choices[0]]
                b = s[choices[1]]

            a_emb = self.embed_doc(a)
            b_emb = self.embed_doc(b)

            l_batch.append(a_emb)
            r_batch.append(b_emb)

            #Negative similarity
            negs = []
            num_negs = 0
            while num_negs < self.k_neg:
                index = np.random.choice(len(docs))
                
                if index != i: #if it's not from the same document
                    neg_doc = docs[index]
                    neg_choice = np.random.choice(len(neg_doc),1)[0]
                    
                    c = neg_doc[neg_choice]
                    c_emb = self.embed_doc(c)
                    
                    negs.append(c_emb)
                    num_negs += 1

            neg_batch.append(torch.stack(negs))
        
        l_batch = torch.stack(l_batch)
        r_batch = torch.stack(r_batch)
        neg_batch = torch.stack(neg_batch)
        
        l_batch = l_batch.unsqueeze(1)
        r_batch = r_batch.unsqueeze(1)
        
        return l_batch, r_batch, neg_batch
