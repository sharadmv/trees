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
    'boar': 'Sus scrofa',
    'bass': 'Micropterus salmoides',
    'bear': 'Ursidae',
    'calf': 'Bos taurus',
    'crab': 'Brachyura',
    'catfish': 'Siluriformes',
    'buffalo': 'Syncerus caffer',
    'clam': 'Bivalvia',
    'cavy': 'Caviidae',
    'dogfish': 'Squalidae',
    'girl': 'Homo sapiens',
    'dolphin': 'Delphinidae',
    'fruitbat': 'Chiroptera',
    'dove': 'Columbiformes',
    'flea': 'Siphonaptera',
    'hawk': 'Accipitridae',
    'herring': 'Clupeidae',
    'lobster': 'Nephropoidea',
    'pheasant': 'Phasianidae',
    'hare': 'Lepus',
    'lark': 'Alaudidae',
    'gull': 'Laridae',
    'moth': 'Lepidoptera',
    'gnat': 'Culex pipiens',
    'seawasp': 'Cubozoa',
    'sealion': 'Otariidae',
    'parakeet': 'Melopsittacus undulatus',
    'pussycat': 'Felis catus',
    'pitviper': 'Viperidae',
    'polecat': 'Mustelinae',
    'seasnake': 'Laticauda',
    'sparrow': 'Emberizinae',
    'slug': 'Gastropoda',
    'pony': 'Equus ferus',
    'sole': 'Soleidae',
    'platypus': 'Ornithorhynchus anatinus',
    'swan': 'Anatidae',
    'piranha': 'Characidae',
    'pike': 'Esox',
    'toad': 'Anura',
    'skua': 'Stercorarius',
    'scorpion': 'Scorpiones',
    'squirrel': 'Sciuridae',
    'worm': 'annelids',
    'wolf': 'Canis lupus',
    'vampire': 'Desmodus rotundus',
    'wasp': 'Hymenoptera',
    'tuna': 'Thunnini',
    'termite': 'Termitidae',
    'vole': 'Arvicolinae',
    'tuatara': 'Hatteria punctata'
}

def convert(a):
    return CONVERSION.get(a, a)

X, y = load('zoo')
tax = api.taxomachine
animals = [convert(a.replace('+', ' ')) for a in y]
ids = []
for animal in tqdm(animals):
    result = tax.TNRS([animal])['results'][0]
    match = result['matches'][0]
    ids.append(match['ot:ottId'])

print ids
if os.path.exists('./tol.nwk'):
    with open('tol.nwk') as fp:
        tol = fp.read()
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
        node = TreeNode({})
        for child in children:
            node.add_child(child)
        return node

final_tree = Tree(root=build_tree(root_node))
with open('zoo.tree', 'wb') as fp:
    pickle.dump(final_tree, fp)
with open('zoo.constraints', 'wb') as fp:
    pickle.dump(list(final_tree.generate_constraints()), fp)
