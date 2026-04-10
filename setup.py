from setuptools import setup, find_packages

setup(
    name='stratoquant',
    version='1.0',
    description='Quant finance library: pricing, greeks, hedging, stats',
    author='Yassine Housseine',
    author_email='yassine.housseine2@gmail.com',
    url='https://github.com/StratoQuantX/stratoquant',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'matplotlib',
        'statsmodels',
        'arch'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License'
    ]
)

# End of file setup.py