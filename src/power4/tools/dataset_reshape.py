from src.deep.MyDataset import MyDataset

db = MyDataset('p4', features={'x', 'y', 's1', 's2'}).load_all(path='0_raw', prefix='p4_') \
    .remove_dupplicates() \
    .split_by_size(size=100000)

for x in db:
    print('saving {}'.format(x.name))
    x.save('1_uniques')

print('done !')
