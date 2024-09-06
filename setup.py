from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT='-e .'  # constant creation
def get_requirements(file_path:str)->List[str]:
    '''
    This function will return list of requirements
    '''
    requirements = []
    with open(file_path, 'r') as file_obj:
        requirements = [line.rstrip('\n') for line in file_obj]
        
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
        
        return requirements

# contains Metadata of project
setup(
    name="MLOpsProject",
    version="0.0.1",
    author="Nitin Luhadiya",
    author_email="nitin.6753@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)