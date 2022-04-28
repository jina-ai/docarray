import argparse
import functools
from time import perf_counter

import h5py

from docarray import Document, DocumentArray
from typing import Iterable

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = perf_counter()
        res = func(*args, **kwargs)
        return (perf_counter() - start, res)

    return wrapper


@timer
def create(da, docs):
    da.extend(docs)


@timer
def read(da, ids):
    da[ids]


@timer
def update(da, docs):
    da[[d.id for d in docs]] = docs


@timer
def delete(da, ids):
    del da[ids]


@timer
def find_by_condition(da, query):
    da.find(query)


@timer
def find_by_vector(da, query, limit):
    return da.find(query, limit=limit)


def get_docs(train):
    return [
        Document(
            id=str(i),
            embedding=x,
            tags={'i': int(i)},
        )
        for i, x in enumerate(train)
    ]


def fmt(value, unit):
    return '{:.3f} {}'.format(value, unit)


def recall(predicted, relevant, eval_at):
    if eval_at == 0:
        return 0.0
    predicted_at_k = predicted[:eval_at]
    n_predicted_and_relevant = len(
        set(predicted_at_k[:, 'id']).intersection(set(relevant[:, 'id']))
    )
    return n_predicted_and_relevant / len(relevant)


def recall_from_numpy(predicted, relevant, eval_at: int):
    """
    >>> recall_from_numpy([1,2,3,4,5], [5,4,3,2,1],5)
    1.0
    >>> recall_from_numpy([1,2,3,4,5], [5,4,3,2,1],4)
    0.8
    >>> recall_from_numpy([1,2,3,4,5], [5,4,3,2,1],3)
    0.3
    >>> recall_from_numpy([1,2,3,4,5], [5,4,3,2,1],2)
    0.4
    >>> recall_from_numpy([1,2,3,4,5], [5,4,3,2,1],1)
    0.2
    """
    if eval_at == 0:
        return 0.0
    predicted_at_k = predicted[:eval_at]
    n_predicted_and_relevant = len(set(predicted_at_k).intersection(set(relevant)))
    return n_predicted_and_relevant / len(relevant)


def save_benchmark_df(benchmark_df):
    benchmark_df.to_csv('benchmark-seconds.csv')

    benchmark_df['Indexing time (C)'] = benchmark_df['Indexing time (C)'].apply(
        lambda value: 1_000_000 / value
    )
    benchmark_df['Query (R)'] = benchmark_df['Query (R)'].apply(lambda value: 1 / value)
    benchmark_df['Update (U)'] = benchmark_df['Update (U)'].apply(
        lambda value: 1 / value
    )
    benchmark_df['Delete (D)'] = benchmark_df['Delete (D)'].apply(
        lambda value: 1 / value
    )
    benchmark_df['Find by vector'] = benchmark_df['Find by vector'].apply(
        lambda value: 1 / value
    )
    benchmark_df['Find by condition'] = benchmark_df['Find by condition'].apply(
        lambda value: 1 / value
    )

    benchmark_df.to_csv('benchmark-qps.csv')


def plot_results(
    find_by_vector_values, storage_backends, create_values, plot_legend=True
):
    find_df = pd.DataFrame(find_by_vector_values)
    find_df.index = [backend for backend, _ in storage_backends]
    find_df = find_df.drop(['sqlite'])
    print('\n\nQuery times')
    print(find_df)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 5))

    find_df.plot(
        kind="bar",
        ax=ax1,
        fontsize=16,
        color=sns.color_palette('muted')[1:4],
        # title='Find by vector per backend and dataset size',
        # ylabel='seconds',
        rot=0,
        legend=plot_legend,
    )
    ax1.set_ylabel('seconds', fontsize=18)
    ax1.set_title('Find by vector per backend', fontsize=18)

    threshold = 0.3
    ax1.hlines(y=threshold, xmin=-20, xmax=20, linewidth=2, color='r', linestyle='--')

    create_df = pd.DataFrame(create_values)
    create_df.index = [backend for backend, _ in storage_backends]

    create_df = create_df.drop(['memory'])
    print('\n\nIndexing times')
    print(create_df)
    create_df.plot(
        kind="bar",
        ax=ax2,
        fontsize=16,
        color=sns.color_palette('muted')[1:4],
        # title='Indexing per backend and dataset size',
        # ylabel='seconds',
        rot=0,
        legend=plot_legend,
    )

    ax2.set_ylabel('seconds', fontsize=18)
    ax2.set_title('Indexing per backend', fontsize=18)

    plt.tight_layout()
    # ax1.legend(fontsize=15)
    # ax2.legend(fontsize=15)
    plt.savefig('benchmark.svg')


def load_sift_dataset(k):
    dataset_path = './sift-128-euclidean.hdf5'
    dataset = h5py.File(dataset_path, 'r')
    train = dataset['train']
    test = dataset['test']

    docs = get_docs(train)
    vector_queries = [x for x in test]
    ground_truth = [x[:k] for x in dataset['neighbors'][0 : len(vector_queries)]]

    return docs, vector_queries, ground_truth


def load_random_dataset(n_dim):
    dataset = np.random.rand(1000_000, n_dim)
    docs = get_docs(dataset)
    vector_queries = [np.random.rand(n_dim) for _ in range(1000)]

    return docs, vector_queries


def get_configuration_storage_backends(n_dim):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--default-hnsw',
        help='Whether to use default HNSW configurations',
        action='store_true',
    )

    args = parser.parse_args()

    if args.default_hnsw:
        storage_backends = [
            ('memory', None),
            ('sqlite', None),
            (
                'annlite',
                {'n_dim': n_dim},
            ),
            ('qdrant', {'n_dim': n_dim, 'scroll_batch_size': 8}),
            ('weaviate', {'n_dim': n_dim}),
            ('elasticsearch', {'n_dim': n_dim}),
        ]
    else:
        storage_backends = [
            ('memory', None),
            ('sqlite', None),
            (
                'annlite',
                {
                    'n_dim': n_dim,
                    'ef_construction': 100,
                    'ef_search': 100,
                    'max_connection': 16,
                },
            ),
            (
                'qdrant',
                {'n_dim': n_dim, 'scroll_batch_size': 8, 'ef_construct': 100, 'm': 16},
            ),
            (
                'weaviate',
                {
                    'n_dim': n_dim,
                    'ef': 100,
                    'ef_construction': 100,
                    'max_connections': 16,
                },
            ),
            ('elasticsearch', {'n_dim': n_dim, 'ef_construction': 100, 'm': 16}),
        ]
    return storage_backends
