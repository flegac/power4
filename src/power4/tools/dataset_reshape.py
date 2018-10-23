from src.deep.MyDataset import MyDataset

db = MyDataset('p4_fat', features={'x', 'y', 's1', 's2'}).load_all(path='build', prefix='p4_') \
    .split_by_size(size=100000)

for x in db:
    print('saving {}'.format(x.name))
    x.save()

print('done !')
