from setuptools import setup, find_packages

setup(
    name='ml-optimizer',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'torch>=1.8.0',
        'numpy>=1.20.0',
        'scikit-learn>=0.24.0',
        'psutil>=5.8.0'
    ],
    author='Your Name',
    description='ML Optimization Library',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    classifiers=[
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)