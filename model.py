import numpy as np
import re
from collections import Counter
from argparser import parser
class Model:
    def __init__(self,window_size=4,embedding_size=100,num_tokens=10000):
        text = open('data.txt', 'r', encoding = 'utf-8').read()
        pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
        self.tokens = pattern.findall(text.lower())[:num_tokens]
        self.window_size = window_size
        self.embedding_size = embedding_size
        self.vocab = list(set(self.tokens))
        self.W1 = np.random.randn(len(self.vocab), self.embedding_size) #shape : (num unique words, embedding size)
        self.W2 = np.random.randn(self.embedding_size, len(self.vocab)) #shape : (embedding size, num unique words)
        self.word_to_index = {word: i for i, word in enumerate(self.vocab)}
        self.index_to_word = {i: word for word, i in self.word_to_index.items()}
        self.distribution = self.generate_distribution()
        
    def generate_distribution(self):
        counts = Counter(self.tokens)
        freqs = np.array([counts[self.index_to_word[i]]**0.75 for i in range(len(self.vocab))])
        return freqs/freqs.sum()
    
    def negative_samples(self,target,k):
        neg = []
        while(len(neg)<k):
            sample = np.random.choice(len(self.vocab),p=self.distribution)
            if(sample!=target):neg.append(sample)
        return neg
            
            
    def generate_training_data(self): 
        y=[]
        for i in range(len(self.tokens)):
            if(i>self.window_size and i<len(self.tokens)-self.window_size):
                context_list = []
                for j in range(i-self.window_size,i+self.window_size+1):
                    if(i!=j):
                        context_list.append(self.word_to_index[self.tokens[j]])     
                y.append((context_list,self.word_to_index[self.tokens[i]]))
        return y
    
    def oneHot(self,index):
        one_hot = np.zeros(len(self.vocab))
        one_hot[index] = 1
        return one_hot
   
    def forward_pass(self,x,target=None,negatives=None):
        m = np.mean(self.W1[x], axis=0) #shape: (1,embedding size)
        if(args.negative_sampling):
            positive_score = self.sigmoid(np.dot(m,self.W2[:,target]))
            negative_scores = [self.sigmoid(np.dot(m,self.W2[:,negs])) for negs in negatives]
            return m, positive_score, negative_scores
        else:
            out = m @ self.W2 #shape(1,num unique words)
            return self.softmax(out) #same shape with probabilites
   
    def cross_entropy_loss(self, predicted, target):
        return -np.sum(target * np.log(predicted + 1e-10))  
    def binary_cross_entropy_loss(self,positive_score,negative_scores):
        return -np.log(positive_score + 1e-10)-sum(np.log(1-neg+1e-10) for neg in negative_scores)
    
    def softmax(self, x):
        exp = np.exp(x - np.max(x))
        return exp / np.sum(exp)
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    
    def back_propagation(self,x ,target,predicted,learning_rate):
        d_out = predicted - target
        d_W2 = np.outer(np.mean(self.W1[x], axis=0), d_out)
        d_m = self.W2 @ d_out
        d_W1 = d_m/len(x)
        self.W2 -= learning_rate * d_W2
        for i in x:
            self.W1[i] -= learning_rate * d_W1
    def back_propagation_neg(self,x,target,negatives,m,positive_score,negative_scores,learning_rate):
        d_true = positive_score-1
        d_negs = list(negative_scores)
        
        d_m = d_true * self.W2[:, target]
        for i,neg in enumerate(negatives):
            d_m = d_m + d_negs[i]*self.W2[:,neg]
            
        self.W2[:,target] -= learning_rate*d_true*m
        for i,neg in enumerate(negatives):
            self.W2[:,neg] -= learning_rate*d_negs[i]*m
        
        d_W1 = d_m/len(x)
        for i in x:
            self.W1[i] -= learning_rate * d_W1
         
    def train(self, epochs=10, learning_rate=0.01):
        data = self.generate_training_data()
        for epoch in range(epochs):
            loss = 0
            for(context, target) in data:
                if(args.negative_sampling):
                    negatives = self.negative_samples(target,args.k)
                    m,positive_score,negative_scores = self.forward_pass(context,target,negatives)
                    loss = loss + self.binary_cross_entropy_loss(positive_score,negative_scores)
                    self.back_propagation_neg(context,target,negatives,m,positive_score,negative_scores,learning_rate)
                else:
                    pred = self.forward_pass(context)
                    target_vec = self.oneHot(target)
                    loss = loss + self.cross_entropy_loss(pred, target_vec)
                    self.back_propagation(context,target_vec,pred,learning_rate)
            print(f'Epoch {epoch+1}, Loss: {loss/len(data)}')
           
    #For testing --------------     
    def analogy(self,a,b,c):
        v1 = self.word_to_index[a]
        v2 = self.word_to_index[b]
        v3 = self.word_to_index[c]
        d = self.W1[v3] - (self.W1[v1]-self.W1[v2])
        similarites = []
        for i,word in self.index_to_word.items():
            cos = np.dot(self.W1[i],d)/(np.linalg.norm(self.W1[i])*np.linalg.norm(d))
            similarites.append((cos,word))
        similarites.sort(key = lambda x : x[0], reverse=True)
        return similarites[:10]
    
    def most_similar(self,word):
        vec = self.W1[self.word_to_index[word]]
        similar = []
        for i,w in self.index_to_word.items():
            if(w==word):continue
            cos = np.dot(self.W1[i],vec)/(np.linalg.norm(self.W1[i])*np.linalg.norm(vec))
            similar.append((cos,w))
        similar.sort(key= lambda x: x[0], reverse = True)
        return similar[:10]
            
                
if __name__ =="__main__":
    args = parser.parse_args()
    model = Model(args.window_size,args.embedding_size,args.num_tokens)
    model.train(args.epochs,args.lr)
    print(model.analogy("man","woman","boy"))
    print(model.most_similar("frankenstein"))
    print(model.most_similar("creature"))
    print(model.most_similar("dog"))
    print(model.most_similar("dead"))
    
    
    
         
    