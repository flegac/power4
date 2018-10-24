from src.deep.MyDataset import MyDataset

features = {'x', 'y'}
training = MyDataset('training', features=features).load_all(path='../../tools/tmp', prefix='p4_fat_')

validation = training.extract('validation', training.size() // 10)

print('training dataset loaded : ', training.size())
print('validation dataset loaded : ', validation.size())
