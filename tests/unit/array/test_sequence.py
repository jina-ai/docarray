import uuid

import pytest
import tempfile

from docarray import Document, DocumentArray
from docarray.array.memory import DocumentArrayInMemory
from docarray.array.qdrant import DocumentArrayQdrant
from docarray.array.sqlite import DocumentArraySqlite
from docarray.array.storage.sqlite import SqliteConfig
from docarray.array.weaviate import DocumentArrayWeaviate
from docarray.array.elastic import DocumentArrayElastic
from docarray.array.clickhouse import DocumentArrayClickHouse
from docarray.array.storage.qdrant import QdrantConfig
from docarray.array.storage.weaviate import WeaviateConfig
from docarray.array.storage.elastic import ElasticConfig
from docarray.array.storage.clickhouse import ClickHouseConfig
import numpy as np

from tests.conftest import tmpfile


@pytest.mark.parametrize(
    'da_cls,config',
    [
        (DocumentArrayInMemory, lambda: None),
        (DocumentArraySqlite, lambda: None),
        (DocumentArrayWeaviate, lambda: WeaviateConfig(n_dim=1)),
        (DocumentArrayQdrant, lambda: QdrantConfig(n_dim=1)),
        (DocumentArrayElastic, lambda: ElasticConfig(n_dim=1)),
        (DocumentArrayClickHouse, lambda: ClickHouseConfig(n_dim=1)),
    ],
)
def test_insert(da_cls, config, start_storage):
    da = da_cls(config=config())
    assert not len(da)
    da.insert(0, Document(text='hello', id="0"))
    da.insert(0, Document(text='world', id="1"))
    assert len(da) == 2
    assert da[0].text == 'world'
    assert da[1].text == 'hello'
    assert da["1"].text == 'world'
    assert da["0"].text == 'hello'


@pytest.mark.parametrize(
    'da_cls,config',
    [
        (DocumentArrayInMemory, lambda: None),
        (DocumentArraySqlite, lambda: None),
        (DocumentArrayWeaviate, lambda: WeaviateConfig(n_dim=1)),
        (DocumentArrayQdrant, lambda: QdrantConfig(n_dim=1)),
        (DocumentArrayElastic, lambda: ElasticConfig(n_dim=1)),
        (DocumentArrayClickHouse, lambda: ClickHouseConfig(n_dim=1)),
    ],
)
def test_append_extend(da_cls, config, start_storage):
    da = da_cls(config=config())
    da.append(Document())
    da.append(Document())
    assert len(da) == 2
    # assert len(da._offset2ids.ids) == 2 will not work unless used in a context manager
    da.extend([Document(), Document()])
    assert len(da) == 4
    # assert len(da._offset2ids.ids) == 4 will not work unless used in a context manager


def update_config_inplace(config, tmpdir, tmpfile):
    variable_names = ['table_name', 'connection', 'collection_name', 'index_name']
    variable_names_db = ['connection']

    for field in variable_names_db:
        if field in config:
            config[field] = str(tmpfile)

    for field in variable_names:
        if field in config:
            config[field] = f'{config[field]}_{uuid.uuid4().hex}'


@pytest.mark.parametrize(
    'storage, config',
    [
        ('sqlite', {'table_name': 'Test', 'connection': 'sqlite'}),
        ('weaviate', {'n_dim': 3, 'name': 'Weaviate'}),
        ('qdrant', {'n_dim': 3, 'collection_name': 'qdrant'}),
        ('elasticsearch', {'n_dim': 3, 'index_name': 'elasticsearch'}),
        ('clickhouse', {'n_dim': 3, 'table_name': 'Test'}),
    ],
)
def test_context_manager_from_disk(storage, config, start_storage, tmpdir, tmpfile):
    config = config
    update_config_inplace(config, tmpdir, tmpfile)

    da = DocumentArray(storage=storage, config=config)

    with da as da_open:
        da_open.append(Document(embedding=np.random.random(3)))
        da_open.append(Document(embedding=np.random.random(3)))

    assert len(da) == 2
    assert len(da._offset2ids.ids) == 2

    da2 = DocumentArray(storage=storage, config=config)

    assert len(da2) == 2
    assert len(da2._offset2ids.ids) == 2

    del da
    del da2
