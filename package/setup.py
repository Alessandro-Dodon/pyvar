from setuptools import setup, find_packages

setup(
    name='PyVaR',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'plotly',
        'scipy',
        'arch'
    ],
    description='VaR and risk modeling toolkit for financial applications',
    author='Alessandro Dodon, Niccolò Lecce, Marco Gasparetti',
)
