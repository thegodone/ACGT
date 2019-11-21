import tensorflow as tf

from libs.layers import feed_forward_net
from libs.layers import GraphConv
from libs.layers import GraphAttn
from libs.layers import GraphGatherReadout
from libs.layers import LinearReadout
from libs.layers import PMAReadout

class NodeEmbedding(tf.keras.layers.Layer):
    def __init__(self, 
                 out_dim, 
                 use_attn,
                 num_heads, 
                 use_ln, 
                 use_ffnn,
                 rate):
        super(NodeEmbedding, self).__init__()

        if use_attn:
            self.gconv = GraphAttn(out_dim, num_heads)
        else:     
            self.gconv = GraphConv(out_dim)

        self.use_ffnn = use_ffnn
        if use_ffnn:
            self.ffnn = feed_forward_net(out_dim)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)            

        self.use_ln = use_ln
        if self.use_ln:
            self.layer_norm1 = tf.keras.layers.LayerNormalization()
            self.layer_norm2 = tf.keras.layers.LayerNormalization()


    def call(self, x, adj, training):            
        h1 = self.gconv(x, adj)
        h1 = self.dropout1(h1, training=training)
        h1 += x
        if self.use_ln:
            h1 = self.layer_norm1(h1)

        if self.use_ffnn:
            h2 = self.ffnn(h1)
            h2 = self.dropout2(h2, training=training)
            h2 += h1
            if self.use_ln:
                h2 = self.layer_norm2(h2)             
        else:
             h2 = h1                

        return h2    

class GraphEmbedding(tf.keras.layers.Layer):
    def __init__(self, 
                 out_dim,
                 readout_method,
                 pooling):
        super(GraphEmbedding, self).__init__()

        if readout_method == 'linear':
            self.readout = LinearReadout(out_dim, pooling)                
        elif readout_method == 'graph_gather':
            self.readout = GraphGatherReadout(out_dim, pooling)                
        elif readout_method == 'pma':
            self.readout = PMAReadout(out_dim, 4)                

    def call(self, x):
        return self.readout(x)   
