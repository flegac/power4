from src.deep.MyDataset import MyDataset
import numpy as np


def current_turn(data: dict):
    s1 = data['s1']
    return np.count_nonzero(s1)


db = MyDataset('p4', features={'x', 'y', 's1', 's2'}).load_all(path='1_uniques', prefix='p4_') \
    .remove_dupplicates() \
    .split_by(key_func=current_turn)

for x in db.values():
    print('saving {}'.format(x.name))
    x.save('2_by_size')

print('done !')
