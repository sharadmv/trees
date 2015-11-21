import json
import cPickle as pickle
from cStringIO import StringIO
import os
from trees.data import load
from tqdm import tqdm
from peyotl.api import APIWrapper
from Bio import Phylo
from trees import TreeNode, TreeLeaf, Tree
api = APIWrapper()

def correct(a):
    return a

CONVERSION = {
    'aardvark': 'Orycteropus afer',
    'boar': 'Sus scrofa',
    'bass': 'Micropterus salmoides',
    'bear': 'Ursidae',
    'calf': 'Bos taurus',
    'cheetah': 'Acinonyx jubatus',
    'chicken': 'Gallus gallus domesticus',
    'chub': 'Leuciscus cephalus',
    'crab': 'Brachyura',
    'catfish': 'Siluriformes',
    'crayfish': 'Astacoidea',
    'buffalo': 'Syncerus caffer',
    'clam': 'Bivalvia',
    'cavy': 'Caviidae',
    'crow': 'Corvus',
    'dogfish': 'Squalidae',
    'girl': 'Homo sapiens',
    'dolphin': 'Delphinidae',
    'fruitbat': 'Chiroptera',
    'dove': 'Columbiformes',
    'elephant': 'Elephantinae',
    'frog': 'Rana catesbeiana',
    'flea': 'Siphonaptera',
    'hawk': 'Accipitridae',
    'herring': 'Clupeidae',
    'haddock': 'Melanogrammus aeglefinus',
    'honeybee': 'Apis',
    'housefly': 'Musca domestica',
    'lobster': 'Nephropoidea',
    'ladybird': 'Coccinellinae',
    'leopard': 'Panthera pardus',
    'pheasant': 'Phasianidae',
    'hare': 'Lepus',
    'lark': 'Alaudidae',
    'gull': 'Laridae',
    'moth': 'Lepidoptera',
    'mole': 'Talpidae',
    'mongoose': 'Herpestidae',
    'newt': 'Pleurodelinae',
    'ostrich': 'Struthio camelus',
    'gnat': 'Culex pipiens',
    'seawasp': 'Cubozoa',
    'sealion': 'Otariidae',
    'parakeet': 'Melopsittacus undulatus',
    'porpoise': 'Phocoenidae',
    'pussycat': 'Felis catus',
    'pitviper': 'Viperidae',
    'polecat': 'Mustelinae',
    'seasnake': 'Laticauda',
    'sparrow': 'Passeridae',
    'slug': 'Gastropoda',
    'raccoon': 'Procyon lotor',
    'reindeer': 'Rangifer tarandus',
    'pony': 'Equus ferus',
    'sole': 'Soleidae',
    'platypus': 'Ornithorhynchus anatinus',
    'seahorse': 'Hippocampus',
    'swan': 'Anatidae',
    'piranha': 'Characidae',
    'pike': 'Esox',
    'toad': 'Anura',
    'skua': 'Stercorarius',
    'scorpion': 'Scorpiones',
    'squirrel': 'Sciuridae',
    'skimmer': 'Charadriiformes',
    'worm': 'annelids',
    'slowworm': 'Anguis fragilis',
    'stingray': 'Hexatrygonidae',
    'wolf': 'Canis lupus',
    'vampire': 'Desmodus rotundus',
    'wasp': 'Hymenoptera',
    'tuna': 'Thunnini',
    'tortoise': 'Testudinidae',
    'termite': 'Termitidae',
    'vole': 'Arvicolinae',
    'tuatara': 'Hatteria punctata',
    'wren': 'Troglodytidae',
}

def convert(a):
    return CONVERSION.get(a, a)

X, y = load('zoo')
if os.path.exists('./ids.pkl'):
    with open('./ids.pkl', 'rb') as fp:
        ids = pickle.load(fp)
else:
    tax = api.taxomachine
    animals = [convert(a.replace('+', ' ')) for a in y]
    ids = []
    for animal in tqdm(animals):
        result = tax.TNRS([animal])['results']
        if len(result) == 0:
            print "Failed:", animal
        result = result[0]
        match = result['matches'][0]
        ids.append(match['ot:ottId'])
    with open('./ids.pkl', 'wb') as fp:
        pickle.dump(ids, fp)

print ids
if os.path.exists('./tol.nwk'):
    with open('tol.nwk') as fp:
        tol = json.load(fp)['newick']
else:
    tol = api.treemachine.induced_subtree(ott_ids=ids)['newick']
tree = Phylo.read(StringIO(tol), 'newick')

root_node = tree.root

def get_idx(name):
    id = int(name.split('_')[-1][3:])
    return ids.index(id)

def build_tree(node):
    if node.is_terminal():
        return TreeLeaf(get_idx(node.name))
    else:
        children = map(build_tree, node.clades)
        node = TreeNode()
        for child in children:
            node.add_child(child)
        return node

final_tree = Tree(root=build_tree(root_node))
with open('zoo.tree', 'wb') as fp:
    pickle.dump(final_tree, fp)
with open('zoo.constraints', 'wb') as fp:
    pickle.dump(list(final_tree.generate_constraints()), fp)
