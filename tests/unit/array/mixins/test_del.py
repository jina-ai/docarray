import pytest

from docarray import DocumentArray, Document
from docarray.array.weaviate import DocumentArrayWeaviate
import numpy as np


@pytest.fixture()
def docs():
    return DocumentArray([Document(id=f'{i}') for i in range(1, 10)])


@pytest.mark.parametrize(
    'to_delete',
    [
        0,
        1,
        4,
        -1,
        list(range(1, 4)),
        [2, 4, 7, 1, 1],
        slice(0, 2),
        slice(2, 4),
        slice(4, -1),
        [True, True, False],
        ...,
    ],
)
def test_del_all(docs, to_delete):
    doc_to_delete = docs[to_delete]
    del docs[to_delete]
    assert doc_to_delete not in docs


@pytest.mark.parametrize(
    'to_delete, missing_id',
    [
        ([True, False], ['1']),
        ([True, True, False], ['1', '2']),
        ([False, True], ['2']),
        ([False, False, True, True], ['3', '4']),
    ],
)
def test_del_boolean_mask(docs, to_delete, missing_id):

    # assert each missing_id is present before deleting
    for m_id in missing_id:
        assert m_id in docs[:, 'id']

    del docs[to_delete]

    # assert each missing_id is NOT present AFTER deleting
    for m_id in missing_id:
        assert m_id not in docs[:, 'id']


@pytest.mark.parametrize(
    ['deleted_ids', 'expected_ids'],
    [
        (['1', '2', '3', '4'], ['5', '6', '7', '8', '9']),
        (['2', '4', '7', '1'], ['3', '5', '6', '8', '9']),
    ],
)
def test_del_by_multiple_idx(docs, deleted_ids, expected_ids):
    del docs[deleted_ids]
    assert docs[:, 'id'] == expected_ids


@pytest.mark.parametrize(
    'da_cls,config,persist',
    [
        (DocumentArrayWeaviate, {'n_dim': 10}, False),
        (DocumentArrayWeaviate, {'name': 'Storage', 'n_dim': 10}, True),
    ],
)
def test_del_da_persist(da_cls, config, persist, docs, start_storage):
    da = da_cls(docs, config=config)
    del da

    da2 = da_cls(config=config)
    if persist:
        assert len(da2) == len(docs)
    else:
        assert len(da2) == 0


def test_del_da_attribute():

    da = DocumentArray(
        [
            Document(embedding=np.array([1, 2, 3]), text='d1'),
            Document(embedding=np.array([1, 2, 3]), text='d2'),
        ]
    )

    q = DocumentArray(
        [
            Document(embedding=np.array([4, 5, 6]), text='q1'),
            Document(embedding=np.array([2, 3, 4]), text='q1'),
        ]
    )

    da.match(q)
    del da[...][:, 'embedding']

    for d in da:
        assert d.embedding is None


def test_del_secondary_index_annlite():

    n_dim = 3
    da = DocumentArray(
        storage='annlite',
        config={'n_dim': n_dim, 'metric': 'Euclidean'},
        secondary_indices_configs={'@c': {'n_dim': 2}},
    )

    with da:
        da.extend(
            [
                Document(
                    id=str(i),
                    embedding=i * np.ones(n_dim),
                    chunks=[
                        Document(id=str(i) + '_0', embedding=[i, i]),
                        Document(id=str(i) + '_1', embedding=[i, i]),
                    ],
                )
                for i in range(10)
            ]
        )

    del da['0']
    assert len(da) == 9
    assert len(da._secondary_indices['@c']) == 18
