import numpy as np
import pytest

from docarray import DocumentArray, Document
from docarray.array.storage.weaviate import WeaviateConfig
from docarray.array.annlite import AnnliteConfig
from docarray.array.qdrant import QdrantConfig
from docarray.array.elastic import ElasticConfig
from docarray.array.clickhouse import ClickHouseConfig


@pytest.fixture
def docs():
    yield (Document(text=str(j)) for j in range(100))


@pytest.fixture
def indices():
    yield (i for i in [-2, 0, 2])


@pytest.mark.parametrize(
    'storage,config',
    [
        ('memory', None),
        ('sqlite', None),
        ('weaviate', WeaviateConfig(n_dim=123)),
        ('annlite', AnnliteConfig(n_dim=123)),
        ('qdrant', QdrantConfig(n_dim=123)),
        ('elasticsearch', ElasticConfig(n_dim=123)),
        ('clickhouse', ClickHouseConfig(n_dim=123)),
    ],
)
def test_getter_int_str(docs, storage, config, start_storage):
    if config:
        docs = DocumentArray(docs, storage=storage, config=config)
    else:
        docs = DocumentArray(docs, storage=storage)
    # getter
    assert docs[99].text == '99'
    assert docs[np.int(99)].text == '99'
    assert docs[-1].text == '99'
    assert docs[0].text == '0'
    # string index
    assert docs[docs[0].id].text == '0'
    assert docs[docs[99].id].text == '99'
    assert docs[docs[-1].id].text == '99'

    with pytest.raises(IndexError):
        docs[100]

    with pytest.raises(KeyError):
        docs['adsad']


@pytest.mark.parametrize(
    'storage,config',
    [
        ('memory', None),
        ('sqlite', None),
        ('weaviate', WeaviateConfig(n_dim=123)),
        ('annlite', AnnliteConfig(n_dim=123)),
        ('qdrant', QdrantConfig(n_dim=123)),
    ],
)
def test_setter_int_str(docs, storage, config, start_storage):
    if config:
        docs = DocumentArray(docs, storage=storage, config=config)
    else:
        docs = DocumentArray(docs, storage=storage)
    # setter
    docs[99] = Document(text='hello')
    docs[0] = Document(text='world')

    assert docs[99].text == 'hello'
    assert docs[-1].text == 'hello'
    assert docs[0].text == 'world'

    docs[docs[2].id] = Document(text='doc2')
    # string index
    assert docs[docs[2].id].text == 'doc2'


@pytest.mark.parametrize(
    'storage,config',
    [
        ('memory', None),
        ('sqlite', None),
        ('weaviate', WeaviateConfig(n_dim=123)),
        ('annlite', AnnliteConfig(n_dim=123)),
        ('qdrant', QdrantConfig(n_dim=123)),
        ('elasticsearch', ElasticConfig(n_dim=123)),
    ],
)
def test_del_int_str(docs, storage, config, start_storage, indices):
    if config:
        docs = DocumentArray(docs, storage=storage, config=config)
    else:
        docs = DocumentArray(docs, storage=storage)
    initial_len = len(docs)
    deleted_elements = 0
    for pos in indices:
        pos_id = docs[pos].id
        del docs[pos]
        deleted_elements += 1
        assert pos_id not in docs
        assert len(docs) == initial_len - deleted_elements

        new_pos_id = docs[pos].id
        new_doc_zero = docs[pos]
        del docs[new_pos_id]
        deleted_elements += 1
        assert len(docs) == initial_len - deleted_elements
        assert pos_id not in docs
        assert new_doc_zero not in docs


@pytest.mark.parametrize(
    'storage,config',
    [
        ('memory', None),
        ('sqlite', None),
        ('weaviate', WeaviateConfig(n_dim=123)),
        ('annlite', AnnliteConfig(n_dim=123)),
        ('qdrant', QdrantConfig(n_dim=123)),
        ('elasticsearch', ElasticConfig(n_dim=123)),
    ],
)
def test_slice(docs, storage, config, start_storage):
    if config:
        docs = DocumentArray(docs, storage=storage, config=config)
    else:
        docs = DocumentArray(docs, storage=storage)
    # getter
    assert len(docs[1:5]) == 4
    assert len(docs[1:100:5]) == 20  # 1 to 100, sep with 5

    # setter
    with pytest.raises(TypeError, match='an iterable'):
        docs[1:5] = Document(text='repl')

    docs[1:5] = [Document(text=f'repl{j}') for j in range(4)]
    for d in docs[1:5]:
        assert d.text.startswith('repl')
    assert len(docs) == 100

    # del
    zero_doc = docs[0]
    twenty_doc = docs[20]
    del docs[0:20]
    assert len(docs) == 80
    assert zero_doc not in docs
    assert twenty_doc in docs


@pytest.mark.parametrize(
    'storage,config',
    [
        ('memory', None),
        ('sqlite', None),
        ('weaviate', WeaviateConfig(n_dim=123)),
        ('annlite', AnnliteConfig(n_dim=123)),
        ('qdrant', QdrantConfig(n_dim=123)),
        ('elasticsearch', ElasticConfig(n_dim=123)),
    ],
)
def test_sequence_bool_index(docs, storage, config, start_storage):
    if config:
        docs = DocumentArray(docs, storage=storage, config=config)
    else:
        docs = DocumentArray(docs, storage=storage)
    # getter
    mask = [True, False] * 50
    assert len(docs[mask]) == 50

    # setter
    mask = [True, False] * 50
    docs[mask, 'text'] = [f'repl{j}' for j in range(50)]

    for idx, d in enumerate(docs):
        if idx % 2 == 0:
            # got replaced
            assert d.text.startswith('repl')
        else:
            assert d.text == str(idx)

    docs[mask] = [Document(text='test') for _ in range(50)]

    for idx, d in enumerate(docs):
        if idx % 2 == 0:
            # got replaced
            assert d.text == 'test'
        else:
            assert d.text == str(idx)

    # del
    del docs[mask]
    assert len(docs) == 50


@pytest.mark.parametrize('nparray', [lambda x: x, np.array, tuple])
@pytest.mark.parametrize(
    'storage,config',
    [
        ('memory', None),
        ('sqlite', None),
        ('weaviate', WeaviateConfig(n_dim=123)),
        ('annlite', AnnliteConfig(n_dim=123)),
        ('qdrant', QdrantConfig(n_dim=123)),
        ('elasticsearch', ElasticConfig(n_dim=123)),
    ],
)
def test_sequence_int(docs, nparray, storage, config, start_storage):
    if config:
        docs = DocumentArray(docs, storage=storage, config=config)
    else:
        docs = DocumentArray(docs, storage=storage)
    # getter
    idx = nparray([1, 3, 5, 7, -1, -2])
    assert len(docs[idx]) == len(idx)

    # setter
    docs[idx] = [Document(text='repl') for _ in range(len(idx))]
    for _id in idx:
        assert docs[_id].text == 'repl'

    # del
    idx = [-3, -4, -5, 9, 10, 11]
    del docs[idx]
    assert len(docs) == 100 - len(idx)

    docs[1, 5, 9] = [Document(text='new') for _ in range(3)]
    assert docs[1].text == 'new'
    assert docs[5].text == 'new'
    assert docs[9].text == 'new'


@pytest.mark.parametrize(
    'storage,config',
    [
        ('memory', None),
        ('sqlite', None),
        ('weaviate', WeaviateConfig(n_dim=123)),
        ('annlite', AnnliteConfig(n_dim=123)),
        ('qdrant', QdrantConfig(n_dim=123)),
        ('elasticsearch', ElasticConfig(n_dim=123)),
    ],
)
def test_sequence_str(docs, storage, config, start_storage):
    if config:
        docs = DocumentArray(docs, storage=storage, config=config)
    else:
        docs = DocumentArray(docs, storage=storage)
    # getter
    idx = [d.id for d in docs[1, 3, 5, 7, -1, -2]]

    assert len(docs[idx]) == len(idx)
    assert len(docs[tuple(idx)]) == len(idx)

    # setter
    docs[idx] = [Document(text='repl') for _ in range(len(idx))]
    idx = [d.id for d in docs[1, 3, 5, 7, -1, -2]]
    for _id in idx:
        assert docs[_id].text == 'repl'

    # del
    idx = [d.id for d in docs[-3, -4, -5, 9, 10, 11]]
    del docs[idx]
    assert len(docs) == 100 - len(idx)


@pytest.mark.parametrize(
    'storage,config',
    [
        ('memory', None),
        ('sqlite', None),
        ('weaviate', WeaviateConfig(n_dim=123)),
        ('annlite', AnnliteConfig(n_dim=123)),
        ('qdrant', QdrantConfig(n_dim=123)),
        ('elasticsearch', ElasticConfig(n_dim=123)),
        ('clickhouse', ClickHouseConfig(n_dim=123)),
    ],
)
def test_docarray_list_tuple(docs, storage, config, start_storage):
    if config:
        docs = DocumentArray(docs, storage=storage, config=config)
    else:
        docs = DocumentArray(docs, storage=storage)
    assert isinstance(docs[99, 98], DocumentArray)
    assert len(docs[99, 98]) == 2


@pytest.mark.parametrize(
    'storage,config',
    [
        ('memory', None),
        ('sqlite', None),
        ('weaviate', WeaviateConfig(n_dim=123)),
        ('annlite', AnnliteConfig(n_dim=123)),
        ('qdrant', QdrantConfig(n_dim=123)),
        ('elasticsearch', ElasticConfig(n_dim=123)),
    ],
)
def test_path_syntax_indexing(storage, config, start_storage):
    da = DocumentArray.empty(3)
    for d in da:
        d.chunks = DocumentArray.empty(5)
        d.matches = DocumentArray.empty(7)
        for c in d.chunks:
            c.chunks = DocumentArray.empty(3)

    if not storage == 'memory':
        if config:
            da = DocumentArray(da, storage=storage, config=config)
        else:
            da = DocumentArray(da, storage=storage)

    assert len(da['@c']) == 3 * 5
    assert len(da['@c:1']) == 3
    assert len(da['@c-1:']) == 3
    assert len(da['@c1']) == 3
    assert len(da['@c-2:']) == 3 * 2
    assert len(da['@c1:3']) == 3 * 2
    assert len(da['@c1:3c']) == (3 * 2) * 3
    assert len(da['@c1:3,c1:3c']) == (3 * 2) + (3 * 2) * 3
    assert len(da['@c 1:3 , c 1:3 c']) == (3 * 2) + (3 * 2) * 3
    assert len(da['@cc']) == 3 * 5 * 3
    assert len(da['@cc,m']) == 3 * 5 * 3 + 3 * 7
    assert len(da['@r:1cc,m']) == 1 * 5 * 3 + 3 * 7


@pytest.mark.parametrize(
    'storage,config',
    [
        ('memory', None),
        ('sqlite', None),
        ('weaviate', WeaviateConfig(n_dim=123)),
        ('annlite', AnnliteConfig(n_dim=123)),
        ('qdrant', QdrantConfig(n_dim=123)),
        ('elasticsearch', ElasticConfig(n_dim=123)),
    ],
)
def test_path_syntax_indexing_set(storage, config, start_storage):
    da = DocumentArray.empty(3)
    for i, d in enumerate(da):
        d.chunks = DocumentArray.empty(5)
        d.matches = DocumentArray([Document(id=f'm{j + (i * 7)}') for j in range(7)])
        for c in d.chunks:
            c.chunks = DocumentArray.empty(3)

    repeat = lambda s, l: [s] * l
    da['@r,c,m,cc', 'text'] = repeat('a', 3 + 5 * 3 + 7 * 3 + 3 * 5 * 3)

    if storage != 'memory':
        if config:
            da = DocumentArray(da, storage=storage, config=config)
        else:
            da = DocumentArray(da, storage=storage)

    assert da['@c', 'text'] == repeat('a', 3 * 5)
    assert da['@c:1', 'text'] == repeat('a', 3)
    assert da['@c-1:', 'text'] == repeat('a', 3)
    assert da['@c1', 'text'] == repeat('a', 3)
    assert da['@c-2:', 'text'] == repeat('a', 3 * 2)
    assert da['@c1:3', 'text'] == repeat('a', 3 * 2)
    assert da['@c1:3c', 'text'] == repeat('a', (3 * 2) * 3)
    assert da['@c1:3,c1:3c', 'text'] == repeat('a', (3 * 2) + (3 * 2) * 3)
    assert da['@c 1:3 , c 1:3 c', 'text'] == repeat('a', (3 * 2) + (3 * 2) * 3)
    assert da['@cc', 'text'] == repeat('a', 3 * 5 * 3)
    assert da['@cc,m', 'text'] == repeat('a', 3 * 5 * 3 + 3 * 7)
    assert da['@r:1cc,m', 'text'] == repeat('a', 1 * 5 * 3 + 3 * 7)
    assert da[0, 'text'] == 'a'
    assert da[[True for _ in da], 'text'] == repeat('a', 3)

    da['@m,cc', 'text'] = repeat('b', 3 + 5 * 3 + 7 * 3 + 3 * 5 * 3)

    assert da['@c', 'text'] == repeat('a', 3 * 5)
    assert da['@c:1', 'text'] == repeat('a', 3)
    assert da['@c-1:', 'text'] == repeat('a', 3)
    assert da['@c1', 'text'] == repeat('a', 3)
    assert da['@c-2:', 'text'] == repeat('a', 3 * 2)
    assert da['@c1:3', 'text'] == repeat('a', 3 * 2)
    assert da['@c1:3c', 'text'] == repeat('b', (3 * 2) * 3)
    assert da['@c1:3,c1:3c', 'text'] == repeat('a', (3 * 2)) + repeat('b', (3 * 2) * 3)
    assert da['@c 1:3 , c 1:3 c', 'text'] == repeat('a', (3 * 2)) + repeat(
        'b', (3 * 2) * 3
    )
    assert da['@cc', 'text'] == repeat('b', 3 * 5 * 3)
    assert da['@cc,m', 'text'] == repeat('b', 3 * 5 * 3 + 3 * 7)
    assert da['@r:1cc,m', 'text'] == repeat('b', 1 * 5 * 3 + 3 * 7)
    assert da[0, 'text'] == 'a'
    assert da[[True for _ in da], 'text'] == repeat('a', 3)

    da[1, 'text'] = 'd'
    assert da[1, 'text'] == 'd'
    assert da[1].text == 'd'

    doc_id = da[0].id
    da[doc_id, 'text'] = 'e'
    assert da[doc_id, 'text'] == 'e'
    assert da[doc_id].text == 'e'

    # setting matches is only possible if the IDs are the same
    da['@m'] = [Document(id=f'm{i}', text='c') for i in range(3 * 7)]
    assert da['@m', 'text'] == repeat('c', 3 * 7)

    # setting by traversal paths with different IDs is not supported
    with pytest.raises(ValueError):
        da['@m'] = [Document() for _ in range(3 * 7)]

    da[2, ['text', 'id']] = ['new_text', 'new_id']
    assert da[2].text == 'new_text'
    assert da[2].id == 'new_id'


@pytest.mark.parametrize('size', [1, 5])
@pytest.mark.parametrize(
    'storage,config_gen',
    [
        ('memory', None),
        ('sqlite', None),
        ('weaviate', lambda: WeaviateConfig(n_dim=123)),
        ('annlite', lambda: AnnliteConfig(n_dim=123)),
        ('qdrant', lambda: QdrantConfig(n_dim=123)),
        ('elasticsearch', lambda: ElasticConfig(n_dim=123)),
    ],
)
def test_attribute_indexing(storage, config_gen, start_storage, size):
    if config_gen:
        da = DocumentArray(storage=storage, config=config_gen())
    else:
        da = DocumentArray(storage=storage)
    da.extend(DocumentArray.empty(size))

    for v in da[:, 'id']:
        assert v
    da[:, 'mime_type'] = [f'type {j}' for j in range(size)]
    for v in da[:, 'mime_type']:
        assert v
    del da[:, 'mime_type']
    for v in da[:, 'mime_type']:
        assert not v

    da[:, ['text', 'mime_type']] = [
        [f'hello {j}' for j in range(size)],
        [f'type {j}' for j in range(size)],
    ]
    da.summary()

    for v in da[:, ['mime_type', 'text']]:
        for vv in v:
            assert vv


@pytest.mark.parametrize(
    'storage', ['memory', 'sqlite', 'weaviate', 'annlite', 'qdrant', 'elasticsearch']
)
def test_tensor_attribute_selector(storage, start_storage):
    import scipy.sparse

    sp_embed = np.random.random([3, 10])
    sp_embed[sp_embed > 0.1] = 0
    sp_embed = scipy.sparse.coo_matrix(sp_embed)

    if storage in ('annlite', 'weaviate', 'qdrant', 'elasticsearch'):
        da = DocumentArray(storage=storage, config={'n_dim': 10})
    else:
        da = DocumentArray(storage=storage)

    da.extend(DocumentArray.empty(3))

    da[:, 'embedding'] = sp_embed

    assert da[:, 'embedding'].shape == (3, 10)

    for d in da:
        assert d.embedding.shape == (1, 10)

    v1, v2 = da[:, ['embedding', 'id']]
    assert isinstance(v1, scipy.sparse.coo_matrix)
    assert isinstance(v2, list)

    v1, v2 = da[:, ['id', 'embedding']]
    assert isinstance(v2, scipy.sparse.coo_matrix)
    assert isinstance(v1, list)


# TODO: since match function is not implemented, this test will
# not work with weaviate storage atm, will be addressed in
# next version
@pytest.mark.parametrize('storage', ['memory', 'sqlite', 'annlite'])
def test_advance_selector_mixed(storage):
    if storage == 'annlite':
        da = DocumentArray(storage=storage, config={'n_dim': 3})
    else:
        da = DocumentArray(storage=storage)

    da.extend(DocumentArray.empty(10))
    da.embeddings = np.random.random([10, 3])

    da.match(da, exclude_self=True)

    assert len(da[:, ('id', 'embedding', 'matches')]) == 3
    assert len(da[:, ('id', 'embedding', 'matches')][0]) == 10


@pytest.mark.parametrize(
    'storage',
    [
        'memory',
        'sqlite',
        'weaviate',
        'annlite',
        'qdrant',
        'elasticsearch',
        'clickhouse',
    ],
)
def test_single_boolean_and_padding(storage, start_storage):
    if storage in ('annlite', 'weaviate', 'qdrant', 'elasticsearch', 'clickhouse'):
        da = DocumentArray(storage=storage, config={'n_dim': 10})
    else:
        da = DocumentArray(storage=storage)
    da.extend(DocumentArray.empty(3))

    with pytest.raises(IndexError):
        da[True]

    with pytest.raises(IndexError):
        da[True] = Document()

    with pytest.raises(IndexError):
        del da[True]

    assert len(da[True, False]) == 1
    assert len(da[False, False, False]) == 0
    assert len(da[True, False, False]) == 1


@pytest.mark.parametrize(
    'storage,config_gen',
    [
        ('memory', None),
        ('sqlite', None),
        ('weaviate', lambda: WeaviateConfig(n_dim=123)),
        ('annlite', lambda: AnnliteConfig(n_dim=123)),
        ('qdrant', lambda: QdrantConfig(n_dim=123)),
        ('elasticsearch', lambda: ElasticConfig(n_dim=123)),
    ],
)
def test_edge_case_two_strings(storage, config_gen, start_storage):
    # getitem
    if config_gen:
        da = DocumentArray(storage=storage, config=config_gen())
    else:
        da = DocumentArray(storage=storage)
    da.extend([Document(id='1'), Document(id='2'), Document(id='3')])
    assert da['1', 'id'] == '1'
    assert len(da['1', '2']) == 2
    assert isinstance(da['1', '2'], DocumentArray)
    with pytest.raises(KeyError):
        da['hello', '2']
    with pytest.raises(AttributeError):
        da['1', 'hello']
    assert len(da['1', '2', '3']) == 3
    assert isinstance(da['1', '2', '3'], DocumentArray)

    # delitem
    del da['1', '2']
    assert len(da) == 1

    if config_gen:
        da = DocumentArray(storage=storage, config=config_gen())
    else:
        da = DocumentArray(storage=storage)
    da.extend([Document(id=str(i), text='hey') for i in range(3)])
    del da['1', 'text']
    assert len(da) == 3
    assert not da[1].text

    with pytest.raises(
        ValueError,
        match='setting the ID of a Document stored in a DocumentArray to None is not allowed',
    ):
        del da['1', 'id']

    del da['2', 'hello']

    # setitem
    if config_gen:
        da = DocumentArray(storage=storage, config=config_gen())
    else:
        da = DocumentArray(storage=storage)
    da.extend([Document(id='1'), Document(id='2'), Document(id='3')])
    da['1', '2'] = DocumentArray.empty(2)
    assert da[0].id != '1'
    assert da[1].id != '2'

    if config_gen:
        da = DocumentArray(
            [Document(id='1'), Document(id='2'), Document(id='3')],
            storage=storage,
            config=config_gen(),
        )
    else:
        da = DocumentArray(
            [Document(id='1'), Document(id='2'), Document(id='3')], storage=storage
        )

    da['1', 'text'] = 'hello'
    assert da['1', 'text'] == 'hello'
    assert da['1'].text == 'hello'

    with pytest.raises(IndexError):
        da['1', 'hellohello'] = 'hello'


@pytest.mark.parametrize(
    'storage,config',
    [
        ('sqlite', None),
        ('weaviate', WeaviateConfig(n_dim=123)),
        ('annlite', AnnliteConfig(n_dim=123)),
        ('qdrant', QdrantConfig(n_dim=123)),
        ('elasticsearch', ElasticConfig(n_dim=123)),
    ],
)
def test_offset2ids_persistence(storage, config, start_storage):
    da = DocumentArray(storage=storage, config=config)

    da.extend(
        [
            Document(id='0'),
            Document(id='2'),
            Document(id='4'),
        ]
    )
    da.insert(1, Document(id='1'))
    da.insert(3, Document(id='3'))

    config = da._config
    da_ids = da[:, 'id']
    assert da_ids == [str(i) for i in range(5)]
    da._persist = True
    da.__del__()

    da = DocumentArray(storage=storage, config=config)

    assert da[:, 'id'] == da_ids


def test_dam_conflicting_ids():
    docs = [
        Document(id='1'),
        Document(id='2'),
        Document(id='3'),
    ]

    d = Document(id='1')
    da = DocumentArray()
    da.extend(docs)
    da.append(d)

    assert len(da) == 4
    assert id(da[0]) == id(docs[0])
    assert id(da[3]) == id(d)

    da[0].text = 'd1'
    da[3].text = 'd2'

    assert docs[0].text == 'd1'
    assert d.text == 'd2'
