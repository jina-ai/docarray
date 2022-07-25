from typing import Optional, overload, TYPE_CHECKING, Dict, Union

from .base import BaseDocumentArray
from .mixins import AllMixins

if TYPE_CHECKING:
    from ..typing import DocumentArraySourceType
    from .memory import DocumentArrayInMemory
    from .sqlite import DocumentArraySqlite
    from .annlite import DocumentArrayAnnlite
    from .weaviate import DocumentArrayWeaviate
    from .elastic import DocumentArrayElastic
    from .clickhouse import DocumentArrayClickHouse
    from .storage.sqlite import SqliteConfig
    from .storage.annlite import AnnliteConfig
    from .storage.weaviate import WeaviateConfig
    from .storage.elastic import ElasticConfig
    from .storage.clickhouse import ClickHouseConfig


class DocumentArray(AllMixins, BaseDocumentArray):
    """
    DocumentArray is a list-like container of :class:`~docarray.Document` objects.

    A DocumentArray can be used to store, embed, and retrieve :class:`~docarray.Document` objects.

    .. code-block:: python

        from docarray import Document, DocumentArray

        da = DocumentArray(
            [Document(text='The cake is a lie'), Document(text='Do a barrel roll!')]
        )
        da.apply(Document.embed_feature_hashing)

        query = Document(text='Can i have some cake?').embed_feature_hashing()
        query.match(da, metric='jaccard', use_scipy=True)

        print(query.matches[:, ('text', 'scores__jaccard__value')])

    .. code-block:: bash

        [['The cake is a lie', 'Do a barrel roll!'], [0.9, 1.0]]

    A DocumentArray can also :ref:`embed its contents using a neural network <embed-via-model>`,
    process them using an :ref:`external Flow or Executor <da-post>`, and persist Documents in a :ref:`Document Store <doc-store>` for
    fast vector search:

    .. code-block:: python

        from docarray import Document, DocumentArray
        import numpy as np

        n_dim = 3
        metric = 'Euclidean'

        # initialize a DocumentArray with ANNLiter Document Store
        da = DocumentArray(
            storage='annlite',
            config={'n_dim': n_dim, 'columns': [('price', 'float')], 'metric': metric},
        )
        # add Documents to the DocumentArray
        with da:
            da.extend(
                [
                    Document(id=f'r{i}', embedding=i * np.ones(n_dim), tags={'price': i})
                    for i in range(10)
                ]
            )
        # perform vector search
        np_query = np.ones(n_dim) * 8
        results = da.find(np_query)

    .. seealso::
        For further details, see our :ref:`user guide <documentarray>`.
    """

    @overload
    def __new__(
        cls, _docs: Optional['DocumentArraySourceType'] = None, copy: bool = False
    ) -> 'DocumentArrayInMemory':
        """Create an in-memory DocumentArray object."""
        ...

    @overload
    def __new__(
        cls,
        _docs: Optional['DocumentArraySourceType'] = None,
        storage: str = 'sqlite',
        config: Optional[Union['SqliteConfig', Dict]] = None,
    ) -> 'DocumentArraySqlite':
        """Create a SQLite-powered DocumentArray object."""
        ...

    @overload
    def __new__(
        cls,
        _docs: Optional['DocumentArraySourceType'] = None,
        storage: str = 'weaviate',
        config: Optional[Union['WeaviateConfig', Dict]] = None,
    ) -> 'DocumentArrayWeaviate':
        """Create a Weaviate-powered DocumentArray object."""
        ...

    @overload
    def __new__(
        cls,
        _docs: Optional['DocumentArraySourceType'] = None,
        storage: str = 'annlite',
        config: Optional[Union['AnnliteConfig', Dict]] = None,
    ) -> 'DocumentArrayAnnlite':
        """Create a AnnLite-powered DocumentArray object."""
        ...

    @overload
    def __new__(
        cls,
        _docs: Optional['DocumentArraySourceType'] = None,
        storage: str = 'elasticsearch',
        config: Optional[Union['ElasticConfig', Dict]] = None,
    ) -> 'DocumentArrayElastic':
        """Create a Elastic-powered DocumentArray object."""
        ...

    @overload
    def __new__(
        cls,
        _docs: Optional['DocumentArraySourceType'] = None,
        storage: str = 'clickhouse',
        config: Optional[Union['ClickHouseConfig', Dict]] = None,
    ) -> 'DocumentArrayClickHouse':
        """Create a ClickHouse-powered DocumentArray object."""
        ...

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        """
        Ensures that offset2ids are stored in the db after
        operations in the DocumentArray are performed.
        """
        self._save_offset2ids()

    def __new__(cls, *args, storage: str = 'memory', **kwargs):
        if cls is DocumentArray:
            if storage == 'memory':
                from .memory import DocumentArrayInMemory

                instance = super().__new__(DocumentArrayInMemory)
            elif storage == 'sqlite':
                from .sqlite import DocumentArraySqlite

                instance = super().__new__(DocumentArraySqlite)
            elif storage == 'annlite':
                from .annlite import DocumentArrayAnnlite

                instance = super().__new__(DocumentArrayAnnlite)
            elif storage == 'weaviate':
                from .weaviate import DocumentArrayWeaviate

                instance = super().__new__(DocumentArrayWeaviate)
            elif storage == 'qdrant':
                from .qdrant import DocumentArrayQdrant

                instance = super().__new__(DocumentArrayQdrant)
            elif storage == 'elasticsearch':
                from .elastic import DocumentArrayElastic

                instance = super().__new__(DocumentArrayElastic)
            elif storage == 'clickhouse':
                from .clickhouse import DocumentArrayClickHouse

                instance = super().__new__(DocumentArrayClickHouse)

            else:
                raise ValueError(f'storage=`{storage}` is not supported.')
        else:
            instance = super().__new__(cls)
        return instance
