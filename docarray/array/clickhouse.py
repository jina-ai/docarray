from .document import DocumentArray
from .storage.clickhouse import StorageMixins, ClickHouseConfig

__all__ = ['DocumentArrayClickHouse', 'ClickHouseConfig']


class DocumentArrayClickHouse(StorageMixins, DocumentArray):
    """This is a :class:`DocumentArray` that uses ClickHouse as
    vector search engine and storage.
    """

    def __new__(cls, *args, **kwargs):
        """``__new__`` method for :class:`DocumentArrayClickHouse`

        :param *args: list of args to instantiate the object
        :param **kwargs: dict of args to instantiate the object
        :return: the instantiated :class:`DocumentArrayClickHouse` object
        """
        return super().__new__(cls)
