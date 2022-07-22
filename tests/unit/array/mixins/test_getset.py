import numpy as np
import pytest
import scipy.sparse
import tensorflow as tf
import torch
from scipy.sparse import csr_matrix

from docarray import DocumentArray, Document
from docarray.array.qdrant import DocumentArrayQdrant
from docarray.array.sqlite import DocumentArraySqlite
from docarray.array.annlite import DocumentArrayAnnlite, AnnliteConfig
from docarray.array.storage.qdrant import QdrantConfig
from docarray.array.storage.weaviate import WeaviateConfig
from docarray.array.weaviate import DocumentArrayWeaviate
from docarray.array.elastic import DocumentArrayElastic, ElasticConfig
from tests import random_docs

rand_array = np.random.random([10, 3])


@pytest.fixture()
def docs():
    rand_docs = random_docs(100)
    return rand_docs


@pytest.fixture()
def nested_docs():
    docs = [
        Document(id='r1', chunks=[Document(id='c1'), Document(id='c2')]),
        Document(id='r2', matches=[Document(id='m1'), Document(id='m2')]),
    ]
    return docs


@pytest.mark.parametrize(
    'storage,config',
    [
        ('memory', {}),
        ('sqlite', {}),
        ('annlite', {'n_dim': 3}),
        ('weaviate', {'n_dim': 3}),
        ('qdrant', {'n_dim': 3}),
        ('elasticsearch', {'n_dim': 3}),
    ],
)
@pytest.mark.parametrize(
    'array',
    [
        rand_array,
        torch.Tensor(rand_array),
        tf.constant(rand_array),
        csr_matrix(rand_array),
    ],
)
def test_set_embeddings_multi_kind(array, storage, config, start_storage):
    da = DocumentArray([Document() for _ in range(10)], storage=storage, config=config)
    da[:, 'embedding'] = array


@pytest.mark.parametrize(
    'da_cls,config',
    [
        (DocumentArray, None),
        (DocumentArraySqlite, None),
        (DocumentArrayAnnlite, AnnliteConfig(n_dim=10)),
        (DocumentArrayWeaviate, WeaviateConfig(n_dim=10)),
        (DocumentArrayQdrant, QdrantConfig(n_dim=10)),
        (DocumentArrayElastic, ElasticConfig(n_dim=10)),
    ],
)
def test_da_get_embeddings(docs, config, da_cls, start_storage):
    if config:
        da = da_cls(config=config)
    else:
        da = da_cls()
    da.extend(docs)
    np.testing.assert_almost_equal(da._get_attributes('embedding'), da.embeddings)
    np.testing.assert_almost_equal(da[:, 'embedding'], da.embeddings)


@pytest.mark.parametrize(
    'da_cls,config',
    [
        (DocumentArray, None),
        (DocumentArraySqlite, None),
        (DocumentArrayAnnlite, AnnliteConfig(n_dim=10)),
        (DocumentArrayWeaviate, WeaviateConfig(n_dim=10)),
        (DocumentArrayQdrant, QdrantConfig(n_dim=10)),
        (DocumentArrayElastic, ElasticConfig(n_dim=10)),
    ],
)
def test_embeddings_setter_da(docs, config, da_cls, start_storage):
    if config:
        da = da_cls(config=config)
    else:
        da = da_cls()
    da.extend(docs)
    emb = np.random.random((100, 10))
    da[:, 'embedding'] = emb
    np.testing.assert_almost_equal(da.embeddings, emb)

    for x, doc in zip(emb, da):
        np.testing.assert_almost_equal(x, doc.embedding)

    da[:, 'embedding'] = None
    if hasattr(da, 'flush'):
        da.flush()
    assert da.embeddings is None or not np.any(da.embeddings)


@pytest.mark.parametrize(
    'da_cls,config',
    [
        (DocumentArray, None),
        (DocumentArraySqlite, None),
        (DocumentArrayAnnlite, AnnliteConfig(n_dim=10)),
        (DocumentArrayWeaviate, WeaviateConfig(n_dim=10)),
        (DocumentArrayQdrant, QdrantConfig(n_dim=10)),
        (DocumentArrayElastic, ElasticConfig(n_dim=10)),
    ],
)
def test_embeddings_wrong_len(docs, config, da_cls, start_storage):
    if config:
        da = da_cls(config=config)
    else:
        da = da_cls()
    da.extend(docs)
    embeddings = np.ones((2, 10))

    with pytest.raises(ValueError):
        da.embeddings = embeddings


@pytest.mark.parametrize(
    'da_cls,config',
    [
        (DocumentArray, None),
        (DocumentArraySqlite, None),
        (DocumentArrayAnnlite, AnnliteConfig(n_dim=10)),
        (DocumentArrayWeaviate, WeaviateConfig(n_dim=10)),
        (DocumentArrayQdrant, QdrantConfig(n_dim=10)),
        (DocumentArrayElastic, ElasticConfig(n_dim=10)),
    ],
)
def test_tensors_getter_da(docs, config, da_cls, start_storage):
    if config:
        da = da_cls(config=config)
    else:
        da = da_cls()
    da.extend(docs)
    tensors = np.random.random((100, 10, 10))
    da.tensors = tensors
    assert len(da) == 100
    np.testing.assert_almost_equal(da.tensors, tensors)

    da.tensors = None
    assert da.tensors is None


@pytest.mark.parametrize(
    'da_cls,config',
    [
        (DocumentArray, None),
        (DocumentArraySqlite, None),
        (DocumentArrayAnnlite, AnnliteConfig(n_dim=10)),
        (DocumentArrayWeaviate, WeaviateConfig(n_dim=10)),
        (DocumentArrayQdrant, QdrantConfig(n_dim=10)),
        (DocumentArrayElastic, ElasticConfig(n_dim=10)),
    ],
)
def test_texts_getter_da(docs, config, da_cls, start_storage):
    if config:
        da = da_cls(config=config)
    else:
        da = da_cls()
    da.extend(docs)
    assert len(da.texts) == 100
    assert da.texts == da[:, 'text']
    texts = ['text' for _ in range(100)]
    da.texts = texts
    assert da.texts == texts

    for x, doc in zip(texts, da):
        assert x == doc.text

    da.texts = None
    if hasattr(da, 'flush'):
        da.flush()

    # unfortunately protobuf does not distinguish None and '' on string
    # so non-set str field in Pb is ''
    assert set(da.texts) == set([''])


@pytest.mark.parametrize(
    'da_cls,config',
    [
        (DocumentArray, None),
        (DocumentArraySqlite, None),
        (DocumentArrayAnnlite, AnnliteConfig(n_dim=10)),
        (DocumentArrayWeaviate, WeaviateConfig(n_dim=10)),
        (DocumentArrayQdrant, QdrantConfig(n_dim=10)),
        (DocumentArrayElastic, ElasticConfig(n_dim=10)),
    ],
)
def test_setter_by_sequences_in_selected_docs_da(docs, config, da_cls, start_storage):
    if config:
        da = da_cls(config=config)
    else:
        da = da_cls()
    da.extend(docs)
    da[[0, 1, 2], 'text'] = 'test'
    assert da[[0, 1, 2], 'text'] == ['test', 'test', 'test']

    da[[3, 4], 'text'] = ['test', 'test']
    assert da[[3, 4], 'text'] == ['test', 'test']

    da[[0], 'text'] = ['jina']
    assert da[[0], 'text'] == ['jina']

    da[[6], 'text'] = ['test']
    assert da[[6], 'text'] == ['test']

    # test that ID not present in da works
    da[[0], 'id'] = '999'
    assert ['999'] == da[[0], 'id']

    da[[0, 1], 'id'] = ['101', '102']
    assert ['101', '102'] == da[[0, 1], 'id']


@pytest.mark.parametrize(
    'da_cls,config',
    [
        (DocumentArray, None),
        (DocumentArraySqlite, None),
        (DocumentArrayAnnlite, AnnliteConfig(n_dim=10)),
        (DocumentArrayWeaviate, WeaviateConfig(n_dim=10)),
        (DocumentArrayQdrant, QdrantConfig(n_dim=10)),
        (DocumentArrayElastic, ElasticConfig(n_dim=10)),
    ],
)
def test_texts_wrong_len(docs, config, da_cls, start_storage):
    if config:
        da = da_cls(config=config)
    else:
        da = da_cls()
    da.extend(docs)
    texts = ['hello']

    with pytest.raises(ValueError):
        da.texts = texts


@pytest.mark.parametrize(
    'da_cls,config',
    [
        (DocumentArray, None),
        (DocumentArraySqlite, None),
        (DocumentArrayAnnlite, AnnliteConfig(n_dim=10)),
        (DocumentArrayWeaviate, WeaviateConfig(n_dim=10)),
        (DocumentArrayQdrant, QdrantConfig(n_dim=10)),
        (DocumentArrayElastic, ElasticConfig(n_dim=10)),
    ],
)
def test_tensors_wrong_len(docs, config, da_cls, start_storage):
    if config:
        da = da_cls(config=config)
    else:
        da = da_cls()
    da.extend(docs)
    tensors = np.ones((2, 10, 10))

    with pytest.raises(ValueError):
        da.tensors = tensors


@pytest.mark.parametrize(
    'da_cls, config',
    [
        (DocumentArray, None),
        (DocumentArraySqlite, None),
        (DocumentArrayAnnlite, AnnliteConfig(n_dim=10)),
        (DocumentArrayWeaviate, WeaviateConfig(n_dim=10)),
        (DocumentArrayQdrant, QdrantConfig(n_dim=10)),
        (DocumentArrayElastic, ElasticConfig(n_dim=10)),
    ],
)
def test_blobs_getter_setter(docs, da_cls, config, start_storage):
    if config:
        da = da_cls(config=config)
    else:
        da = da_cls()
    da.extend(docs)
    with pytest.raises(ValueError):
        da.blobs = [b'cc', b'bb', b'aa', b'dd']

    da.blobs = [b'aa'] * len(da)
    assert da.blobs == [b'aa'] * len(da)

    da.blobs = None
    if hasattr(da, 'flush'):
        da.flush()

    # unfortunately protobuf does not distinguish None and '' on string
    # so non-set str field in Pb is ''
    assert set(da.blobs) == set([b''])


@pytest.mark.parametrize(
    'da_cls, config',
    [
        (DocumentArray, None),
        (DocumentArraySqlite, None),
        (DocumentArrayAnnlite, AnnliteConfig(n_dim=10)),
        (DocumentArrayWeaviate, WeaviateConfig(n_dim=10)),
        (DocumentArrayQdrant, QdrantConfig(n_dim=10)),
        (DocumentArrayElastic, ElasticConfig(n_dim=10)),
    ],
)
def test_ellipsis_getter(nested_docs, da_cls, config, start_storage):
    if config:
        da = da_cls(config=config)
    else:
        da = da_cls()
    da.extend(nested_docs)
    flattened = da[...]
    assert len(flattened) == 6
    for d, doc_id in zip(flattened, ['c1', 'c2', 'r1', 'm1', 'm2', 'r2']):
        assert d.id == doc_id


@pytest.mark.parametrize(
    'da_cls, config',
    [
        (DocumentArray, None),
        (DocumentArraySqlite, None),
        (DocumentArrayAnnlite, AnnliteConfig(n_dim=10)),
        (DocumentArrayWeaviate, WeaviateConfig(n_dim=10)),
        (DocumentArrayQdrant, QdrantConfig(n_dim=10)),
        (DocumentArrayElastic, ElasticConfig(n_dim=10)),
    ],
)
def test_ellipsis_attribute_setter(nested_docs, da_cls, config, start_storage):
    if config:
        da = da_cls(config=config)
    else:
        da = da_cls()
    da.extend(nested_docs)
    da[..., 'text'] = 'new'
    assert all(d.text == 'new' for d in da[...])


@pytest.mark.parametrize(
    'da_cls, config',
    [
        (DocumentArray, None),
        (DocumentArraySqlite, None),
        (DocumentArrayAnnlite, AnnliteConfig(n_dim=6)),
        (DocumentArrayWeaviate, WeaviateConfig(n_dim=6)),
        (DocumentArrayElastic, ElasticConfig(n_dim=6)),
    ],
)
def test_zero_embeddings(da_cls, config, start_storage):
    a = np.zeros([10, 6])
    if config:
        da = da_cls.empty(10, config=config)
    else:
        da = da_cls.empty(10)

    # all zero, dense
    da[:, 'embedding'] = a
    np.testing.assert_almost_equal(da.embeddings, a)
    for d in da:
        assert d.embedding.shape == (6,)

    # all zero, sparse
    sp_a = scipy.sparse.coo_matrix(a)
    da[:, 'embedding'] = sp_a
    np.testing.assert_almost_equal(da.embeddings.todense(), sp_a.todense())
    for d in da:
        # scipy sparse row-vector can only be a (1, m) not squeezible
        assert d.embedding.shape == (1, 6)

    # near zero, sparse
    a = np.random.random([10, 6])
    a[a > 0.1] = 0
    sp_a = scipy.sparse.coo_matrix(a)
    da[:, 'embedding'] = sp_a
    np.testing.assert_almost_equal(da.embeddings.todense(), sp_a.todense())
    for d in da:
        # scipy sparse row-vector can only be a (1, m) not squeezible
        assert d.embedding.shape == (1, 6)


def test_getset_secondary_index_annlite():

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
                for i in range(3)
            ]
        )
    with da:
        da[0] = Document(
            embedding=-1 * np.ones(n_dim),
            chunks=[
                Document(id='c_0', embedding=[-1, -1]),
                Document(id='c_1', embedding=[-2, -2]),
            ],
        )

    assert (da[0].embedding == -1 * np.ones(n_dim)).all()
    assert (da[0].chunks[0].embedding == [-1, -1]).all()
    assert (da[0].chunks[1].embedding == [-2, -2]).all()

    assert (da._secondary_indices['@c']['c_0'].embedding == [-1, -1]).all()
    assert (da._secondary_indices['@c']['c_1'].embedding == [-2, -2]).all()
