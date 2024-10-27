import numpy as np
from pprint import pprint


lines = open('t.txt', 'r').readlines()
perfs = [eval(x.split('|')[0]) for x in lines]
times = [eval(x.split('|')[1]) for x in lines]

datasets = open('run.txt', 'r').readline().split(':')

perfs = np.array(perfs).reshape((10, 30, 4))
idx = np.argmax(perfs[:, :, 0], axis=-1)
res = perfs[np.arange(perfs.shape[0]), idx, :]

print('\t\tf1\t\tpd\tpf\tprec')
pprint(list(zip(datasets, res)))
