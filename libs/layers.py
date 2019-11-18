import tensorflow as tf

def feed_forward_net(dim):
    return tf.keras.Sequential([
            tf.keras.layers.Dense(4*dim, activation=tf.nn.relu),
            tf.keras.layers.Dense(dim)
    ])

class GraphConv(tf.keras.layers.Layer):
    def __init__(self, out_dim, **kwargs):
        super(GraphConv, self).__init__(**kwargs)

        self.dense = tf.keras.layers.Dense(units=out_dim, use_bias=False)
        self.act = tf.nn.relu

    def call(self, x, adj, act=tf.nn.relu):
        h = self.dense(x)
        h = tf.matmul(adj, h)
        h = self.act(h)
        return h

class GraphAttn(tf.keras.layers.Layer):
    def __init__(self, out_dim, num_heads, **kwargs):
        super(GraphAttn, self).__init__(**kwargs)

        assert out_dim % num_heads == 0
    
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.depth = out_dim // num_heads

        self.wq = tf.keras.layers.Dense(out_dim, use_bias=False)
        self.wk = tf.keras.layers.Dense(out_dim, use_bias=False)
        self.wv = tf.keras.layers.Dense(out_dim, use_bias=False)
        self.dense = tf.keras.layers.Dense(units=out_dim, use_bias=False)

        self.act = tf.nn.relu

    def multi_head_attn(self, xq, xk, xv, adj):
        matmul_qk = tf.matmul(xq, xk, transpose_b=True)
        scale = tf.cast(tf.shape(xk)[-1], tf.float32)

        adj = tf.tile(tf.expand_dims(adj, 1), self.num_heads)
        matmul_qk = matmul_qk / tf.math.sqrt(scale)
        attn_adj = tf.multiply(matmul_qk, adj)
        attn_adj = tf.tanh(attn_adj)
        out = tf.matmul(attn_adj, xv)
        return out

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))        
        return tf.transpose(x, perm=[0,2,1,3])

    def call(self, x, adj):
        batch_size = tf.shape(x)[0]        

        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        xq = self.split_heads(xq, batch_size)
        xk = self.split_heads(xk, batch_size)
        xv = self.split_heads(xv, batch_size)

        h = self.multi_head_attention(xq, xk, xv, adj)
        h = tf.reshape(h, (batch_size, -1, self.out_dim))
        h = self.dense(concat_attention)
        h = self.act(h)
        return h

class LinearReadout(tf.keras.layers.Layer):        
    def __init__(self, out_dim, **kwargs):
        super(LinearReadout, self).__init__(**kwargs)

        self.dense = tf.keras.layers.Dense(units=out_dim, use_bias=False)
        self.act = tf.nn.sigmoid

    def call(self, x):
        z = self.dense(x)
        z = tf.math.reduce_sum(z, axis=1)
        return self.act(z)
    
class GraphGatherReadout(tf.keras.layers.Layer):        
    def __init__(self, out_dim, **kwargs):
        super(GraphGatherReadout, self).__init__(**kwargs)

        self.dense1 = tf.keras.layers.Dense(units=out_dim, use_bias=False)
        self.dense2 = tf.keras.layers.Dense(units=out_dim, use_bias=False)
        self.act = tf.nn.sigmoid

    def call(self, x):
        z = tf.multiply(tf.nn.sigmoid(self.dense1(x)), self.dense2(x))
        z = tf.math.reduce_sum(z)
        return self.act(z)
                    
    
