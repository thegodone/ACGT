import tensorflow as tf

from libs.layers import feed_forward_net
from libs.layers import GraphConv
from libs.layers import GraphAttn
from libs.layers import GraphGatherReadout
from libs.layers import LinearReadout

class NodeEmbedding(tf.keras.layers.Layer):
    def __init__(self, 
                 out_dim, 
                 use_attn=True,
                 num_heads=4, 
                 use_ln=True, 
                 use_ffn=True,
                 rate=0.1):
        super(NodeEmbedding, self).__init__()

        if use_attn:
            self.gconv = GraphAttn(out_dim, num_heads)
        else:     
            self.gconv = GraphConv(out_dim)

        if use_ffn:
            self.ffn = feed_forward_net(out_dim)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)            

        self.use_ln = use_ln
        if self.use_ln:
            self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
            self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)


    def call(self, x, adj, training):            
        h1 = self.gconv(x, adj)
        h1 = self.dropout1(h1, training=training)
        h1 += x
        if self.use_ln:
            h1 = self.layer_norm1(h1)

        if use_ffn:
            h2 = self.ffn(h1)
            h2 = self.dropout2(h2, training=training)
            h2 += h1
            if self.use_ln:
                h2 = self.layer_norm(h2)             
        else:
             h2 = h1                
        return h2    

class GraphEmbedding(tf.keras.layers.Layer):
    def __init__(self, 
                 out_dim,
                 readout_method):
        super(GraphEmbedding, self).__init__()

        if readout_method == 'linear':
            self.readout = LinearReadout(out_dim)                
        if readout_method == 'graph_gather':
            self.readout = GraphGatherReadout(out_dim)                

    def call(self, x):
        return self.readout(x)            
