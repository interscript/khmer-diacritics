from symspellpy import SymSpell, Verbosity
import numpy as np
import pandas as pd
import os
from tqdm import tqdm


sym_spell = SymSpell()
corpus_path = 'examples/spell-check/plain.txt'
sym_spell.create_dictionary(corpus_path)


f = open('examples/spell-check/input.txt')
data_input = f.read().split('\n')
if data_input[-1] == '': data_input = data_input[:-1]
f.close()

f = open('examples/spell-check/target.txt')
data_target = f.read().split('\n')
if data_target[-1] == '': data_target = data_target[:-1]
f.close()


count = 0
for inp, t in tqdm(zip(data_input, data_target), total=len(data_input)):
    suggestions = sym_spell.lookup(t, Verbosity.CLOSEST,
                               max_edit_distance=2, include_unknown=True)
    count += suggestions[0].term == inp
print(count/len(data_input))