import csv

import tensorflow as tf
import numpy as np

from rdkit import Chem

def read_csv(prop, s_name, l_name):
    with open('./data/'+prop+'.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        contents = np.asarray([(row[s_name], row[l_name]) for row in reader])
    return contents

def atom_feature(atom):

    def one_of_k_encoding(x, allowable_set):
        if x not in allowable_set:
            raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
        return list(map(lambda s: float(x == s), allowable_set))


    def one_of_k_encoding_unk(x, allowable_set):
        """Maps inputs not in the allowable set to the last element."""
        if x not in allowable_set:
            x = allowable_set[-1]
        return list(map(lambda s: float(x == s), allowable_set))

    return np.array(
        one_of_k_encoding_unk(atom.GetSymbol(),
            ['C', 'N', 'O', 'S', 'F', 'H', 'Si', 'P', 'Cl', 'Br',
            'Li', 'Na', 'K', 'Mg', 'Ca', 'Fe', 'As', 'Al', 'I', 'B',
            'V', 'Tl', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn',
            'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'Mn', 'Cr', 'Pt', 'Hg', 'Pb']) +
        one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
        one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
        one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) +
        [atom.GetIsAromatic()]
    )    


def convert_smiles_to_graph(smi_and_label):    
    smi = smi_and_label[0].numpy()
    label = float(smi_and_label[1])
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        adj = Chem.rdmolops.GetAdjacencyMatrix(mol)
        feature = [atom_feature(atom) for atom in mol.GetAtoms()]
        return [feature, adj, label]
 
 
def get_dataset(prop, 
                batch_size,
                max_atoms=100,
                train_ratio=0.8, 
                supervised=True,
                shuffle=True, 
                buffer_size=None, 
                seed=123):

    if buffer_size is None:
        buffer_size = batch_size*10


    s_name = 'smiles'
    l_name = 'p_np'
    smi_and_label = read_csv(prop, s_name, l_name)
    total_ds = tf.data.Dataset.from_tensor_slices(smi_and_label)
    total_ds = total_ds.map(
        lambda x: tf.py_function(func=convert_smiles_to_graph, 
                                 inp=[x], 
                                 Tout=[tf.float32, tf.float32, tf.float32]))
    total_ds = total_ds.apply(tf.data.experimental.ignore_errors())
    if shuffle:
        total_ds = total_ds.shuffle(
            buffer_size=buffer_size, 
            seed=seed
        )
    num_total = smi_and_label.shape[0]
    num_train = int(num_total*train_ratio)

    train_ds = total_ds.take(num_train)
    train_ds = train_ds.shuffle(buffer_size)
    train_ds = train_ds.padded_batch(batch_size, padded_shapes=([None, 58], [None,None], []))
    train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

    test_ds = total_ds.skip(num_train)
    test_ds = test_ds.padded_batch(batch_size, padded_shapes=([None, 58], [None,None], []))
    test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)

    return train_ds, test_ds 
