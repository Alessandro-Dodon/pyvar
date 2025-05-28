from setuptools import setup, find_packages

setup(
    name='pyvar',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.22,<2.0',
        'pandas>=1.4',
        'scipy>=1.8',
        'yfinance>=0.2',
        'plotly>=5.0',
        'kaleido>=0.2',
        'matplotlib>=3.4',
        'seaborn>=0.11',
        'arch>=7.0',
        'statsmodels>=0.12',
        'ipython>=8.13,<9.0',
        'requests>=2.31'
    ],
    description='PyVaR: A financial risk modeling and VaR analysis toolkit',
    author='Alessandro Dodon, Niccolò Lecce, Marco Gasparetti',
    license='MIT',
    python_requires='>=3.7'
)
