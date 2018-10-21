from src.deep.dataset import Dataset

db = Dataset('p4_fat', features={'X', 'Y'}).load_all(path='tmp', prefix='p4_') \
    .rename('X', 'x') \
    .rename('Y', 'y') \
    .split_by_size(size=100000)

for x in db:
    x.save()

print('done !')
