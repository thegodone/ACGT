import argparse
import sys
import os

from absl import logging
from absl import app

import numpy as np
import tensorflow as tf

from libs.predict_net import PredictNet
from libs.lr_scheduler import WarmUpSchedule

FLAGS = None


def train_single_step():

    return


def train(model, optimizer, dataset):

    model_name = FLAGS.prefix
    model_name += '_' + FLAGS.prop
    model_name += '_' + str(FLAGS.num_layers)
    model_name += '_' + str(FLAGS.node_dim)
    model_name += '_' + str(FLAGS.graph_dim)
    model_name += '_' + str(FLAGS.use_attn)
    model_name += '_' + str(FLAGS.num_heads)
    model_name += '_' + str(FLAGS.use_ln)
    model_name += '_' + str(FLAGS.use_ffnn)
    model_name += '_' + str(FLAGS.dropout_rate)
    model_name += '_' + str(FLAGS.readout_method)
    ckpt_path = './save/'+model_name

    checkpoint = tf.train.Checkpoint(
        model=model, 
        optimizer=optimizer
    )
    ckpt_manager = tf.train.CheckpointManager(
        checkpoint=checkpoint, 
        directory=ckpt_path, 
        max_to_keep=FLAGS.max_to_keep
    )

    start_epoch = 0
    if ckpt_manager.latest_checkpoint:
        start_epoch = int(ckpt_manager.latest_checkpoint.split('_')[-1])

    for epoch in range(start_epoch, FLAGS.num_epoches):
        st = time.time()
        total_loss = 0.0

        for (batch, (x, adj, label)) in enumerate(dataset):
            batch_loss, t_loss = train_step(x, adj, label)

        ckpt_manager.save()            

    return


def main(_):

    model = PredictNet(
        num_layers=FLAGS.num_layers,
        node_dim=FLAGS.node_dim,
        graph_dim=FLAGS.graph_dim,
        use_attn=FLAGS.use_attn,
        num_heads=FLAGS.num_heads,
        use_ln=FLAGS.use_ln,
        use_ffnn=FLAGS.use_ffnn,
        dropout_rate=FLAGS.dropout_rate,
        readout_method=FLAGS.readout_method
    )

    lr_scheulder = None
    if FLAGS.lr_schedule == 'warmup':
        lr_scheduler = WarmUpSchedule(
            d_model=FLAGS.graph_dim, 
            warmup_steps=FLAGS.warmup_steps
        )
    elif FLAGS.lr_schedule == 'stair':
        lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=FLAGS.init_lr,
            decay_steps=FLAGS.decay_steps,
            decay_rate=FLAGS.decay_rate,
            staircase=True,
            name='Stair learning rate decay'
        )            

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr_scheduler,
        beta_1=FLAGS.beta_1,
        beta_2=FLAGS.beta_2,
        epsilon=FLAGS.opt_epsilon
    )    

    metrics = [
        tf.keras.metrics.Accuracy(),
        tf.keras.metrics.AUC(curve='ROC'),
        tf.keras.metrics.AUC(curve='PR'),
        tf.keras.metrics.Precision(top_k=FLAGS.top_k),
        tf.keras.metrics.Recall(top_k=FLAGS.top_k),
    ]

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=metrics
    )
    model.summary()
    train(model, dataset)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', type=str, default='GTA', 
                        help='Prefix for this training')
    parser.add_argument('--prop', type=str, default='BACE', 
                        help='Target property to train')

    # Hyper-parameters for model construction
    parser.add_argument('--num_layers', type=int, default=4, 
                        help='Number of node embedding layers')
    parser.add_argument('--node_dim', type=int, default=64, 
                        help='Dimension of node embeddings')
    parser.add_argument('--graph_dim', type=int, default=256, 
                        help='Dimension of a graph embedding')
    parser.add_argument('--use_attn', type=bool, default=True, 
                        help='Whether to use multi-head attentions')
    parser.add_argument('--num_heads', type=int, default=256, 
                        help='Number of attention heads')
    parser.add_argument('--use_ln', type=bool, default=True, 
                        help='Whether to use layer normalizations')
    parser.add_argument('--use_ffnn', type=bool, default=True, 
                        help='Whether to use feed-forward nets')
    parser.add_argument('--dropout_rate', type=float, default=0.1, 
                        help='Dropout rates in node embedding layers')
    parser.add_argument('--readout_method', type=str, default='linear', 
                        help='Readout method to be used')

    # Hyper-parameters for training
    parser.add_argument('--lr_schedule', type=str, default='warmup', 
                        help='How to schedule learning rate')
    parser.add_argument('--init_lr', type=float, default=1e-3, 
                        help='Initial learning rate,\
                              Do not need for warmup scheduling')
    parser.add_argument('--beta_1', type=float, default=0.9, 
                        help='Beta1 in adam optimizer')
    parser.add_argument('--beta_2', type=float, default=0.98, 
                        help='Beta2 in adam optimizer')
    parser.add_argument('--opt_epsilon', type=float, default=1e-9, 
                        help='Epsilon in adam optimizer')
    parser.add_argument('--warmup_steps', type=int, default=4000, 
                        help='Warmup steps for warmup scheduling')
    parser.add_argument('--decay_steps', type=int, default=10000, 
                        help='Decay steps for stair learning rate scheduling')
    parser.add_argument('--decay_rate', type=float, default=0.1, 
                        help='Decay rate for stair learning rate scheduling')
    parser.add_argument('--max_to_keep', type=int, default=5, 
                        help='Maximum number of checkpoint files to be kept')

    # Hyper-parameters for evaluation
    parser.add_argument('--top_k', type=int, default=None,
                       help='Top-k instances for evaluating Precision or Recall')
    FLAGS, unparsed = parser.parse_known_args()
    app.run(main=main, argv=[sys.argv[0]] + unparsed)

