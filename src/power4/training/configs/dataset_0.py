from src.deep.dataset import Dataset

features = {'x', 'y'}
training = Dataset('training', features=features).load_all(path='../data', prefix='p4_0000')

validation = Dataset('validation', features=features).load_all(path='../data', prefix='p4_0001')

print('training dataset loaded : ', len(training.x))
print('validation dataset loaded : ', len(validation.x))
