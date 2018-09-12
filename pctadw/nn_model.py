from keras.models import Sequential, Model
from keras.layers import Input, Dense, Embedding, Flatten, Activation, Multiply, RepeatVector, Masking, concatenate
from keras.utils import to_categorical
from keras import backend as K
from keras import regularizers
import keras
import gc
import numpy as np

def PCTADW_2(num_node, vocabulary, vector_dim, alpha):

    x_input = Input(shape=(1,))

    xparent = Embedding(num_node, vector_dim, input_length=1, name='parent_embedding', embeddings_regularizer=regularizers.l2(alpha))(x_input)

    xparent = Flatten()(xparent)

    xchild = Embedding(num_node, vector_dim, input_length=1, name='child_embedding', embeddings_regularizer=regularizers.l2(alpha))(x_input)

    xchild = Flatten()(xchild)

    x = concatenate([xparent, xchild])
    
    xeword = Input(shape=(1,))
    x = Multiply()([Flatten()(RepeatVector(2*vector_dim)(xeword)), x])
    x = Masking()(x)
    output_word = Dense(vocabulary, activation='softmax', name = 'outputword', kernel_regularizer=regularizers.l2(alpha))(x)

    xeparent = Input(shape=(1,))
    xparent = Multiply()([Flatten()(RepeatVector(vector_dim)(xeparent)), xparent])
    xparent = Masking()(xparent)
    
    output_parent = Dense(num_node, activation='softmax', name = 'outputparent', kernel_regularizer=regularizers.l2(alpha))(xparent)
    
    xechild = Input(shape=(1,))
    xchild = Multiply()([Flatten()(RepeatVector(vector_dim)(xechild)), xchild])
    xchild = Masking()(xchild)

    output_child = Dense(num_node, activation='softmax', name = 'outputchild', kernel_regularizer=regularizers.l2(alpha))(xchild)

    
    model = Model(inputs=[x_input,xeparent,xechild, xeword], outputs=[output_parent, output_child, output_word ])
    
    
    return model

def PCTADW_1(num_node, vocabulary, vector_dim, alpha):
    
    x_input = Input(shape=(1,))

    x = Embedding(num_node, vector_dim, input_length=1, name='input_embedding', embeddings_regularizer=regularizers.l2(alpha))(x_input)
    x = Flatten()(x)
    
    xeword = Input(shape=(1,))
    xword = Multiply()([Flatten()(RepeatVector(vector_dim)(xeword)), x])
    xword = Masking()(xword)
    output_word = Dense(vocabulary, activation='softmax', name = 'outputword', kernel_regularizer=regularizers.l2(alpha))(xword)

    xeparent = Input(shape=(1,))
    xparent = Multiply()([Flatten()(RepeatVector(vector_dim)(xeparent)), x])
    xparent = Masking()(xparent)
    
    output_parent = Dense(num_node, activation='softmax', name = 'outputparent', kernel_regularizer=regularizers.l2(alpha))(xparent)
    
    xechild = Input(shape=(1,))
    xchild = Multiply()([Flatten()(RepeatVector(vector_dim)(xechild)), x])
    xchild = Masking()(xchild)

    output_child = Dense(num_node, activation='softmax', name = 'outputchild', kernel_regularizer=regularizers.l2(alpha))(xchild)

    
    model = Model(inputs=[x_input,xeparent,xechild, xeword], outputs=[output_parent, output_child, output_word ])
    
    
    return model

def train(data, num_epochs = 100, m = 5, vector_dim = 64, alpha = 0.0, batchsize=1024, window_size = 2, training_weight = {'outputparent':1.0, 'outputword':1.0, 'outputchild':1.0}, split_sample_size = 50000, model_name = 'PCTADW-2'):
    
    num_node = data.node_size 
    vocabulary = data.vocabulary_size
    epoch_num = 1
    
    if model_name == 'PCTADW-1':
        
        model = PCTADW_1(num_node, vocabulary, vector_dim, alpha)
        
    elif model_name == 'PCTADW-2':
        
        model = PCTADW_2(num_node, vocabulary, vector_dim, alpha)
    
    else:
        raise Exception('Incorrect model name.')
        
    model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'], loss_weights=training_weight)
        
    for i in range(num_epochs): 
        
        input_doc, input_ep, input_ec, input_ew, output_p, output_c, output_w = data.get_samples(window_size=window_size, m=m)
        
        
        size =  input_ep.shape[0]
        
        # split the samples
        if size > split_sample_size:
            
            num = int(size/split_sample_size) + 1
            
            size1 = int(size/num) + 1
    
            for j in range(num):
                
                start_idx = j*size1
                end_idx = (j+1)*size1
        
                if end_idx < size:
                    
                    model.fit([input_doc[start_idx:end_idx], input_ep[start_idx:end_idx], input_ec[start_idx:end_idx], input_ew[start_idx:end_idx]], [to_categorical(output_p[start_idx:end_idx], num_node), to_categorical(output_c[start_idx:end_idx], num_node), to_categorical(output_w[start_idx:end_idx], vocabulary)], epochs=epoch_num, batch_size=batchsize)
                    
                else:
                    
                    model.fit([input_doc[start_idx:], input_ep[start_idx:], input_ec[start_idx:], input_ew[start_idx:]], [to_categorical(output_p[start_idx:], num_node), to_categorical(output_c[start_idx:], num_node), to_categorical(output_w[start_idx:], vocabulary)], epochs=epoch_num, batch_size=batchsize)
                    
        else:            
           
            model.fit([input_doc, input_ep, input_ec, input_ew], [to_categorical(output_p, num_node), to_categorical(output_c, num_node), to_categorical(output_w, vocabulary)], epochs=epoch_num, batch_size=batchsize)
        
        gc.collect()
        
    if model_name == 'PCTADW-1':
        
        return model.layers[5].get_weights()[0]
    
    elif model_name == 'PCTADW-2':
        
        return np.concatenate([model.layers[2].get_weights()[0], model.layers[4].get_weights()[0]], axis=1)
        
