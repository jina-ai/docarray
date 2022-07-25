import pytest

from docarray import Document
from docarray.array.memory import DocumentArrayInMemory
from docarray.array.elastic import DocumentArrayElastic, ElasticConfig
from docarray.array.qdrant import DocumentArrayQdrant
from docarray.array.sqlite import DocumentArraySqlite
from docarray.array.annlite import DocumentArrayAnnlite, AnnliteConfig
from docarray.array.storage.qdrant import QdrantConfig
from docarray.array.weaviate import DocumentArrayWeaviate, WeaviateConfig
from docarray.array.elastic import DocumentArrayElastic, ElasticConfig
from docarray.array.clickhouse import DocumentArrayClickHouse, ClickHouseConfig


@pytest.mark.parametrize(
    'da_cls,config',
    [
        (DocumentArrayInMemory, None),
        (DocumentArraySqlite, None),
        (DocumentArrayAnnlite, AnnliteConfig(n_dim=128)),
        (DocumentArrayWeaviate, WeaviateConfig(n_dim=128)),
        (DocumentArrayQdrant, QdrantConfig(n_dim=128)),
        (DocumentArrayElastic, ElasticConfig(n_dim=128)),
        (DocumentArrayClickHouse, ClickHouseConfig(n_dim=128)),
    ],
)
def test_construct_docarray(da_cls, config, start_storage):
    if config:
        da = da_cls(config=config)
        assert len(da) == 0

        da = da_cls(Document(), config=config)
        assert len(da) == 1

        da = da_cls([Document(), Document()], config=config)
        assert len(da) == 2

        da = da_cls((Document(), Document()), config=config)
        assert len(da) == 2

        da = da_cls((Document() for _ in range(10)), config=config)
        assert len(da) == 10
    else:
        da = da_cls()
        assert len(da) == 0

        da = da_cls(Document())
        assert len(da) == 1

        da = da_cls([Document(), Document()])
        assert len(da) == 2

        da = da_cls((Document(), Document()))
        assert len(da) == 2

        da = da_cls((Document() for _ in range(10)))
        assert len(da) == 10

        if da_cls is DocumentArrayInMemory:
            da1 = da_cls(da)
            assert len(da1) == 10


@pytest.mark.parametrize(
    'da_cls,config',
    [
        (DocumentArrayInMemory, None),
        (DocumentArraySqlite, None),
        (DocumentArrayAnnlite, AnnliteConfig(n_dim=128)),
        (DocumentArrayWeaviate, WeaviateConfig(n_dim=128)),
        (DocumentArrayQdrant, QdrantConfig(n_dim=128)),
        (DocumentArrayElastic, ElasticConfig(n_dim=128)),
        (DocumentArrayClickHouse, ClickHouseConfig(n_dim=128)),
    ],
)
@pytest.mark.parametrize('is_copy', [True, False])
def test_docarray_copy_singleton(da_cls, config, is_copy, start_storage):
    d = Document()
    if config:
        da = da_cls(d, copy=is_copy, config=config)
    else:
        da = da_cls(d, copy=is_copy)

    d.id = 'hello'
    if da_cls == DocumentArrayInMemory:
        if is_copy:
            assert da[0].id != 'hello'
        else:
            assert da[0].id == 'hello'
    else:
        assert da[0].id != 'hello'


@pytest.mark.parametrize(
    'da_cls,config',
    [
        (DocumentArrayInMemory, None),
        (DocumentArraySqlite, None),
        (DocumentArrayAnnlite, AnnliteConfig(n_dim=128)),
        (DocumentArrayWeaviate, WeaviateConfig(n_dim=128)),
        (DocumentArrayQdrant, QdrantConfig(n_dim=128)),
        (DocumentArrayElastic, ElasticConfig(n_dim=128)),
        (DocumentArrayClickHouse, ClickHouseConfig(n_dim=128)),
    ],
)
@pytest.mark.parametrize('is_copy', [True, False])
def test_docarray_copy_da(da_cls, config, is_copy, start_storage):
    d1 = Document()
    d2 = Document()
    if config:
        da = da_cls([d1, d2], copy=is_copy, config=config)
    else:
        da = da_cls([d1, d2], copy=is_copy)
    d1.id = 'hello'
    if da_cls == DocumentArrayInMemory:
        if is_copy:
            assert da[0].id != 'hello'
        else:
            assert da[0].id == 'hello'
    else:
        assert da[0] != 'hello'


@pytest.mark.parametrize(
    'da_cls,config',
    [
        (DocumentArrayInMemory, None),
        (DocumentArraySqlite, None),
        (DocumentArrayAnnlite, AnnliteConfig(n_dim=1)),
        (DocumentArrayQdrant, QdrantConfig(n_dim=1)),
        (DocumentArrayElastic, ElasticConfig(n_dim=128)),
        (DocumentArrayClickHouse, ClickHouseConfig(n_dim=128)),
    ],
)
@pytest.mark.parametrize('is_copy', [True, False])
def test_docarray_copy_list(da_cls, config, is_copy, start_storage):
    d1 = Document()
    d2 = Document()
    da = da_cls([d1, d2], copy=is_copy, config=config)
    d1.id = 'hello'
    if da_cls == DocumentArrayInMemory:
        if is_copy:
            assert da[0].id != 'hello'
        else:
            assert da[0].id == 'hello'
    else:
        assert da[0] != 'hello'
