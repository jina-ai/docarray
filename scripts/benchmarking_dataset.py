import random

import pandas as pd

from docarray import DocumentArray
from rich.console import Console
from rich.table import Table
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
    get_configuration_storage_backends,
    load_sift_dataset,
    load_sift_dataset,
)


def run_benchmark(
    storage_backends, n_index_values, index_docs, vector_queries, ground_truth
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

    for idx, n_index in enumerate(n_index_values):
        docs = index_docs[:n_index]

        console.print(f'Reading dataset')
        query_docs = random.sample([d.id for d in docs], n_query)
        docs_to_delete = random.sample(docs, n_query)
        docs_to_update = random.sample(docs, n_query)
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
                read_time, _ = read(da, query_docs)
                console.print(f'\tupdating {n_query} docs ...')
                update_time, _ = update(da, docs_to_update)
                console.print(
                    f'\tfinding {n_query} docs by vector averaged {len(vector_queries)} times ...'
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

                console.print(f'\tdeleting {n_query} docs ...')
                delete_time, _ = delete(da, [d.id for d in docs_to_delete])
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
    n_query = 1
    D = 128
    K = 10
    # Benchmark
    storage_backends = get_configuration_storage_backends(D)
    docs, vector_queries, ground_truth = load_sift_dataset(K)
    n_index_values = [len(docs)]

    find_by_vector_values, create_values, benchmark_df = run_benchmark(
        storage_backends, n_index_values, docs, vector_queries, ground_truth
    )
    plot_results(
        find_by_vector_values, storage_backends, create_values, plot_legend=False
    )
    save_benchmark_df(benchmark_df)
