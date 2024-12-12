from setuptools import setup, find_packages

setup(
    name='featureutils',
    version='1.0',
    author='Jona Ruthardt',
    author_email='jona@ruthardt.de',
    description='Feature management utility for PyTorch objects.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/JonaRuthardt/FeatureUtils',
    packages=find_packages(),
    classifiers=[
        'License :: OSI Approved :: Apache Software License 2.0',
        'Programming Language :: Python :: 3',
    ],
    python_requires='>=3.6',
    install_requires=[
        'torch',
        'tqdm',
        'portalocker',
    ],
    include_package_data=True,
    zip_safe=False,
)