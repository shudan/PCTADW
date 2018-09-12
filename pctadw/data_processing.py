import random
import numpy as np
import networkx as nx
from random import shuffle

def file_processing(textfile_path, edgefile_path):
    
    with open(textfile_path,'r') as f:
        
        '''
        Builds a vocabulary and encode the text associated with nodes by the vocabulary.

        Store the data in the class of DataInfo.
        
        '''
        
        textdata = f.readlines()
        
        word_to_index = {}
        vocabulary = set()
        
        for d in textdata:
            
            vocabulary.update(d.split())
      
        for j, w in enumerate(vocabulary):
            
            word_to_index[w] = j
            
        
        text = [ [word_to_index[w] for w in d.split()] for d in textdata]
        
    
    with open(edgefile_path,'r') as f:
        
        edges = np.loadtxt(f, dtype=int)
    
    data = DataInfo(len(vocabulary), text, edges)
    
    return data


class DataInfo():
    
    def __init__(self, vocabulary_size, desc, edgesdata ):
        
        self.desc = desc
        
        self.vocabulary_size = vocabulary_size
        
        self.graph = nx.DiGraph()
        
        self.graph.add_edges_from(edgesdata)
        
        self.node_size = len(self.desc)
        
    
    @staticmethod
    def neigh_and_weight(node, window_size, func):
        
        """
        Finds the neighbors of the node within the number of window_size layers.
        
        """
        
        neighs = []
        weights = []
        
        layer_nodes = [node]
        layer_weights = [1]
        
        for i in range(window_size):
            
            tmp_neighs = []
            
            tmp_weights = []
            
            for j, n in enumerate(layer_nodes):
                    
                tmp_nodes = list(func(n))
                    
                tmp_len = len(tmp_nodes)
                    
                if tmp_len:
                   
                    tmp_weights.extend([layer_weights[j]/tmp_len]*tmp_len)
                   
                    tmp_neighs.extend(tmp_nodes)
                    
            layer_nodes = tmp_neighs
            layer_weights = tmp_weights
            
            if not layer_nodes:
                break
                
            neighs.extend(layer_nodes)
            weights.extend(layer_weights)
            
        return neighs, weights
    
    @staticmethod
    def sampling(pop, k, empty_idx, weights=None):
        
        if pop:
            
            samples = random.choices(pop, weights=weights, k=k)
            existence_index = [1]*k
        
        else:
            
            samples = [empty_idx]*k
            existence_index = [0]*k
            
        return samples, existence_index    
            
        
    def get_samples(self, window_size = 1, m = 5): 
        
        """
        Generates samples for each epoch of training.
        
        """
        
        # if the node doesn't exist
        empty_node = 0
        
        empty_word = 0
        
        nodes = list(self.graph.nodes())
        
        shuffle(nodes)
        
        samples = []
        
        for n in nodes:
            
            children, children_weight = DataInfo.neigh_and_weight(n, window_size, self.graph.predecessors)
            parents, parents_weight = DataInfo.neigh_and_weight(n, window_size, self.graph.successors)
            words = self.desc[n]
            
            k_num = max(len(children), len(parents), 1)
            
            k_num = min(k_num, m)
            
           
            # sampling
            child_pred, child_ex = DataInfo.sampling(children, k_num, empty_node, weights=children_weight)
            parent_pred, parent_ex = DataInfo.sampling(parents, k_num, empty_node, weights=parents_weight)
            word_pred, word_ex = DataInfo.sampling(words, k_num, empty_word)
                
            
            samples.extend((n, p_ex, c_ex, w_ex, p_pred, c_pred, w) for p_ex, c_ex, w_ex, p_pred, c_pred, w in zip(parent_ex, child_ex, word_ex, parent_pred, child_pred, word_pred))
            
        samples = np.array(samples)
        np.random.shuffle(samples)
            
        return ( samples[:,i] for i in range(samples.shape[1]) )  
