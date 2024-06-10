from setuptools import setup, find_packages

setup(
    name="torchTrainigLoop",
    version="0.1.0",
    author='tathagata',
    author_email='tathagatadasworkspace@gmail.com',
    description='It make torch training loop more simple and visualize loss in curve.',
    packages=find_packages(),
    install_requires=[
        'torch>=2.3.0',
        'numpy>=1.26.3',
        'ipython>=8.18.1',
        'jupyter>=1.0.0',
        'plotly>=5.22.0',
    ]
)
