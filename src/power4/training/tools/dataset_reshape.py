from src.deep.dataset import Dataset

db = Dataset('p4_fat').load_all(path='../data/positions', prefix='p4_').split_by_size(size=100000)

for x in db:
    x.save()

print('done !')
