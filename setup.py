import sys
from os import path

from setuptools import find_packages
from setuptools import setup

if sys.version_info < (3, 7, 0):
    raise OSError(f'DocArray requires Python >=3.7, but yours is {sys.version}')

try:
    pkg_name = 'docarray'
    libinfo_py = path.join(pkg_name, '__init__.py')
    libinfo_content = open(libinfo_py, 'r', encoding='utf8').readlines()
    version_line = [l.strip() for l in libinfo_content if l.startswith('__version__')][
        0
    ]
    exec(version_line)  # gives __version__
except FileNotFoundError:
    __version__ = '0.0.0'

try:
    with open('README.md', encoding='utf8') as fp:
        _long_description = fp.read()
except FileNotFoundError:
    _long_description = ''

setup(
    name=pkg_name,
    packages=find_packages(),
    version=__version__,
    include_package_data=True,
    description='The data structure for unstructured data',
    author='Jina AI',
    author_email='hello@jina.ai',
    license='Apache 2.0',
    url='https://github.com/jina-ai/docarray',
    download_url='https://github.com/jina-ai/docarray/tags',
    long_description=_long_description,
    long_description_content_type='text/markdown',
    zip_safe=False,
    setup_requires=['setuptools>=18.0', 'wheel'],
    install_requires=['numpy', 'rich>=12.0.0'],
    extras_require={
        # req usage, please see https://docarray.jina.ai/#install
        'common': [
            'protobuf>=3.21.0',
            'lz4',
            'requests',
            'matplotlib',
            'Pillow',
            'fastapi',
            'uvicorn',
        ],
        'full': [
            'protobuf>=3.21.0',
            'lz4',
            'requests',
            'matplotlib',
            'Pillow',
            'trimesh',
            'scipy',
            'av',
            'fastapi',
            'uvicorn',
            'strawberry-graphql',
            'weaviate-client~=3.3.0',
            'annlite>=0.3.2',
            'qdrant-client~=0.7.3',
            'elasticsearch>=8.2.0',
        ],
        'qdrant': [
            'qdrant-client~=0.7.3',
        ],
        'annlite': [
            'annlite>=0.3.2',
        ],
        'weaviate': [
            'weaviate-client~=3.3.0',
        ],
        'elasticsearch': [
            'elasticsearch>=8.2.0',
        ],
        'test': [
            'pytest',
            'pytest-timeout',
            'pytest-mock',
            'pytest-cov',
            'pytest-repeat',
            'pytest-reraise',
            'mock',
            'pytest-custom_exit_code',
            'black==22.3.0',
            'tensorflow==2.7.0',
            'paddlepaddle==2.2.0',
            'torch==1.9.0',
            'torchvision==0.10.0',
            'datasets',
            'onnx',
            'onnxruntime',
            'jupyterlab',
            'transformers>=4.16.2',
            'weaviate-client~=3.3.0',
            'annlite>=0.3.2',
            'elasticsearch>=8.2.0',
            'jina',
        ],
    },
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Unix Shell',
        'Environment :: Console',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Topic :: Database :: Database Engines/Servers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Internet :: WWW/HTTP :: Indexing/Search',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Multimedia :: Video',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    project_urls={
        'Documentation': 'https://docarray.jina.ai',
        'Source': 'https://github.com/jina-ai/docarray/',
        'Tracker': 'https://github.com/jina-ai/docarray/issues',
    },
    keywords='docarray deep-learning data-structures cross-modal multi-modal unstructured-data nested-data neural-search',
)
