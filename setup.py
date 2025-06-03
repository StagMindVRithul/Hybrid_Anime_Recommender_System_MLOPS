from setuptools import setup, find_packages

with open('requirements.txt',encoding='utf-8') as f:
    requirements = f.read().splitlines()

setup(
    name='mlops-anime-recommendation',
    version='0.1.0',
    author='V Rithul',
    description='A MLOps project for Anime Recommendation System',
    packages=find_packages(),
    install_requires=requirements,
    python_requires='>=3.11'
)