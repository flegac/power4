from setuptools import setup

setup(
    name='power4',
    version='',
    packages=['src', 'src.deep', 'src.games', 'src.games.mcts', 'src.games.state', 'src.games.negamax', 'src.power4',
              'src.power4.tools', 'src.power4.training', 'src.power4.training.data', 'src.power4.training.models',
              'src.power4.training.configs', 'src.power4.training.versions'],
    url='',
    license='',
    author='Flo',
    author_email='',
    description='', install_requires=['keras', 'matplotlib', 'tensorflow', 'numpy']
)
