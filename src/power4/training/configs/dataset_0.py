from src.deep.dataset import Dataset

training = Dataset('training').load_all(path='../data', prefix='p4_0000')
validation = Dataset('validation').load_all(path='../data', prefix='p4_0001')

print('training dataset loaded : ', len(training.x))
print('validation dataset loaded : ', len(validation.x))
