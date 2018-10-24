from src.deep.MyDataset import MyDataset

db = MyDataset('p4_fat', features={'x', 'y', 's1', 's2'}).load_all(path='0_raw', prefix='p4_') \
    .remove_dupplicates() \
    .shuffle()


print('done !')
