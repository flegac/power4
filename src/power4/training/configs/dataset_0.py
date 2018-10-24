from src.deep.MyDataset import MyDataset

features = {'x', 'y'}
training = MyDataset('training', features=features).load_all(path='../../tools/1_uniques', prefix='p4_')

validation = training.extract('validation', training.size() // 10)

print('training dataset loaded : ', training.size())
print('validation dataset loaded : ', validation.size())
