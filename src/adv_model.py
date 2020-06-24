import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from sklearn.feature_extraction.text import CountVectorizer

class Embedder():
    def __init__(self, d_embed=None, input_embedding=None, vocab=None):
        self.vocab = vocab
        self.weights = input_embedding
        self.d_embed = d_embed
        
        # Set Device
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
            
        if input_embedding is not None:
            self.weights.to(self.device)
        
    def train_vocab(self, train, min_df=5, max_df=.99):
        """ Builds the vocabulary object and initiates weights for the embedder """
        Vectorizer = CountVectorizer(min_df = min_df,max_df = max_df)
        Vectorizer.fit(train)
        self.vocab = Vectorizer.vocabulary_
        
        self.weights = nn.Embedding(len(self.vocab), self.d_embed, max_norm=10)
        self.weights.to(self.device)
        
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
        output = torch.sum(self.weights(d),dim=0)
        #output[output != output] = 0 #necessary for documents with all unseen vocabs

        output.to(self.device)

        if normalize:
            output = output / output.norm()
        return output


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        outputs = self.linear(x)
        return outputs


class Discriminator():
    def __init__(self, classifier, optimizer):
        self.classifier = classifier
        self.opt = opt


class StarSpaceAdv(nn.Module):
    def __init__(self, input_embedder, lr = .01, k_neg = 3):
        super(StarSpaceAdv, self).__init__()

        self.embedder = input_embedder
        self.k_neg = k_neg
        self.opt = torch.optim.Adam([self.embedder.weights.weight], lr=lr)
        
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
                    
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

            a_emb = self.embedder.embed_doc(a)
            b_emb = self.embedder.embed_doc(b)

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
                    c_emb = self.embedder.embed_doc(c)
                    
                    negs.append(c_emb)
                    num_negs += 1

            neg_batch.append(torch.stack(negs))
        
        l_batch = torch.stack(l_batch)
        r_batch = torch.stack(r_batch)
        neg_batch = torch.stack(neg_batch)
        
        l_batch = l_batch.unsqueeze(1)
        r_batch = r_batch.unsqueeze(1)
        
        return l_batch, r_batch, neg_batch
