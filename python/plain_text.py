with open('examples/spell-check/input.txt', 'r') as f:
    txt = ' '.join(f.read().split('\n'))
with open('examples/spell-check/plain.txt', 'w') as f:
    f.write(txt)