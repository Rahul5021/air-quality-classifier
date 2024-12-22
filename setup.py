from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'
def get_requirements(filepath:str)->List[str]:
    '''
    this function will return the list of requirements
    '''
    requirements = []
    with open(filepath, mode='r') as f:
        requirements = f.readlines()
        requirements = [req.replace("\n", "")for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    
    return requirements

setup(
    name = 'air-quality',
    version = '0.0.1',
    author = 'rahul',
    author_email= 'agrawal.rahul.1025@gmail.com',
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')
)