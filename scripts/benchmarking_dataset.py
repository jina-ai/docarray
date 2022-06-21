import argparse
import random

import pandas as pd

from docarray import DocumentArray
from rich.console import Console
from rich.table import Table
import h5py
import numpy as np

from benchmarking_utils import (
    create,
    read,
    update,
    delete,
    find_by_condition,
    find_by_vector,
    get_docs,
    fmt,
    recall_from_numpy,
    save_benchmark_df,
    plot_results,
)


def get_configuration_storage_backends(argparse):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--default-hnsw',
        help='Whether to use default HNSW configurations',
        action='store_true',
    )

    args = parser.parse_args()

    if args.default_hnsw:
        storage_backends = [
                ('weaviate', {'n_dim': D, 'distance': 'l2-squared'}),
            (
                'annlite',
                {'n_dim': D},
            ),
            ('qdrant', {'n_dim': D, 'scroll_batch_size': 8}),
            ('elasticsearch', {'n_dim': D}),
            ('sqlite', None),
            ('memory', None),
        ]
    else:
        storage_backends = [
            (
                'weaviate',
                {'n_dim': D, 'ef': 100, 'ef_construction': 100, 'max_connections': 16, 'distance': 'l2-squared'},
            ),
            (
                'annlite',
                {
                    'n_dim': D,
                    'ef_construction': 100,
                    'ef_search': 100,
                    'max_connection': 16,
                },
            ),
            (
                'qdrant',
                {'n_dim': D, 'scroll_batch_size': 8, 'ef_construct': 100, 'm': 16},
            ),
            ('elasticsearch', {'n_dim': D, 'ef_construction': 100, 'm': 16}),
            ('sqlite', None),
            ('memory', None),
        ]
    return storage_backends


def run_benchmark(
    X_tr,
    X_te,
    dataset,
    n_index_values,
    n_vector_queries,
    n_query,
    storage_backends,
    K,
    D,
):
    table = Table(
        title=f'DocArray Benchmarking n_index={n_index_values[-1]} n_query={n_query} D={D} K={K}'
    )
    benchmark_df = pd.DataFrame(
        {
            'Storage Backend': [],
            'Indexing time (C)': [],
            'Query (R)': [],
            'Update (U)': [],
            'Delete (D)': [],
            'Find by vector': [],
            f'Recall at k={K} for vector search': [],
            'Find by condition': [],
        }
    )

    for col in benchmark_df.columns:
        table.add_column(col)

    console = Console()
    find_by_vector_values = {str(n_index): [] for n_index in n_index_values}
    create_values = {str(n_index): [] for n_index in n_index_values}

    console.print(f'Reading dataset')
    docs = get_docs(X_tr)
    docs_to_delete = random.sample(docs, n_query)
    docs_to_update = random.sample(docs, n_query)
    vector_queries = [x for x in X_te]
    ground_truth = [x[0:K] for x in dataset['neighbors'][0 : len(vector_queries)]]

    for idx, n_index in enumerate(n_index_values):
        for backend, config in storage_backends:
            try:
                console.print('\nBackend:', backend.title())
                # for n_i in n_index:
                if not config:
                    da = DocumentArray(storage=backend)
                else:
                    da = DocumentArray(storage=backend, config=config)
                console.print(f'\tindexing {n_index} docs ...')
                create_time, _ = create(da, docs)
                # for n_q in n_query:
                console.print(f'\treading {n_query} docs ...')
                read_time, _ = read(
                    da,
                    random.sample([d.id for d in docs], n_query),
                )
                console.print(f'\tupdating {n_query} docs ...')
                update_time, _ = update(da, docs_to_update)
                console.print(f'\tdeleting {n_query} docs ...')
                delete_time, _ = delete(da, [d.id for d in docs_to_delete])
                console.print(
                    f'\tfinding {n_query} docs by vector averaged {n_vector_queries} times ...'
                )
                if backend == 'memory':
                    find_by_vector_time, aux = find_by_vector(
                        da, vector_queries[0], limit=K
                    )
                    recall_at_k = 1
                elif backend == 'sqlite':
                    find_by_vector_time, result = find_by_vector(
                        da, vector_queries[0], limit=K
                    )
                    recall_at_k = 1
                else:
                    recall_at_k_values = []
                    find_by_vector_times = []
                    for i, query in enumerate(vector_queries):
                        find_by_vector_time, results = find_by_vector(
                            da, query, limit=K
                        )
                        find_by_vector_times.append(find_by_vector_time)
                        recall_at_k_values.append(
                            recall_from_numpy(
                                np.array(results[:, 'tags__i']), ground_truth[i], K
                            )
                        )

                    recall_at_k = np.mean(recall_at_k_values)
                    find_by_vector_time = np.mean(find_by_vector_times)

                console.print(f'\tfinding {n_query} docs by condition ...')
                find_by_condition_time, _ = find_by_condition(
                    da, {'tags__i': {'$eq': 0}}
                )
                if idx == len(n_index_values) - 1:
                    table.add_row(
                        backend.title(),
                        fmt(create_time, 's'),
                        fmt(read_time * 1000, 'ms'),
                        fmt(update_time * 1000, 'ms'),
                        fmt(delete_time * 1000, 'ms'),
                        fmt(find_by_vector_time, 's'),
                        '{:.3f}'.format(recall_at_k),
                        fmt(find_by_condition_time, 's'),
                    )
                    benchmark_df.append(
                        pd.DataFrame(
                            [
                                [
                                    backend.title(),
                                    create_time,
                                    read_time,
                                    update_time,
                                    delete_time,
                                    find_by_vector_time,
                                    recall_at_k,
                                    find_by_condition_time,
                                ]
                            ],
                            columns=benchmark_df.columns,
                        )
                    )

                # store find_by_vector time
                find_by_vector_values[str(n_index)].append(find_by_vector_time)
                create_values[str(n_index)].append(create_time)
                console.print(table)
                da.clear()
                del da
            except Exception as e:
                console.print(f'Storage Backend {backend} failed: {e}')

        console.print(table)
        return find_by_vector_values, create_values, benchmark_df


if __name__ == "__main__":

    # Parameters settable by the user
    n_query = 1
    K = 10
    DATASET_PATH = 'sift-128-euclidean.hdf5'
    np.random.seed(123)

    # Variables gathered from the dataset
    dataset = h5py.File(DATASET_PATH, 'r')
    X_tr = dataset['train']
    X_te = dataset['test']
    D = X_tr.shape[1]
    n_index_values = [len(X_tr)]
    n_vector_queries = len(X_te)

    # Benchmark
    storage_backends = get_configuration_storage_backends(argparse)
    find_by_vector_values, create_values, benchmark_df = run_benchmark(
        X_tr,
        X_te,
        dataset,
        n_index_values,
        n_vector_queries,
        n_query,
        storage_backends,
        K,
        D,
    )
    plot_results(
        find_by_vector_values, storage_backends, create_values, plot_legend=False
    )
    save_benchmark_df(benchmark_df)
