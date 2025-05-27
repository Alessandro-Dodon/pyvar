from setuptools import setup, find_packages

setup(
    name='pyvar',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy<2.0',          # ⬅️ Restrict to NumPy < 2.0
        'pandas',
        'scipy',
        'yfinance',
        'plotly',
        'kaleido',
        'matplotlib',
        'seaborn',
        'arch',
        'statsmodels',
        'ipython',
        'requests'
    ],
    description='PyVaR: A financial risk modeling and VaR analysis toolkit',
    author='Alessandro Dodon, Niccolò Lecce, Marco Gasparetti',
    license='MIT',
    python_requires='>=3.7'
)
