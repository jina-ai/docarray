from typing import MutableSequence, Iterable, Iterator, Union
from docarray import Document

from qdrant_client import QdrantClient

from docarray.array.storage.base.seqlike import BaseSequenceLikeMixin


class SequenceLikeMixin(BaseSequenceLikeMixin):
    @property
    def client(self) -> QdrantClient:
        raise NotImplementedError()

    @property
    def collection_name(self) -> str:
        raise NotImplementedError()

    @property
    def config(self):
        raise NotImplementedError()

    def __eq__(self, other):
        """Compare this object to the other, returns True if and only if other
        as the same type as self and other has the same meta information

        :param other: the other object to check for equality
        :return: ``True`` if other is equal to self
        """
        # two DAW are considered as the same if they have the same client meta data
        return (
            type(self) is type(other)
            and self.client.openapi_client.client.host
            == other.openapi_client.client.host
            and self.config == other.config
        )

    def __len__(self):
        return self.client.http.collections_api.get_collection(
            self.collection_name
        ).result.vectors_count

    def __contains__(self, x: Union[str, 'Document']):
        if isinstance(x, str):
            return self._id_exists(x)
        elif isinstance(x, Document):
            return self._id_exists(x.id)
        else:
            return False

    def _id_exists(self, x: str):
        try:
            self._get_doc_by_id(x)
            return True
        except KeyError:
            return False

    def __repr__(self):
        return f'<DocumentArray[Qdrant] (length={len(self)}) at {id(self)}>'

    def __bool__(self):
        """To simulate ```l = []; if l: ...```

        :return: returns true if the length of the array is larger than 0
        """
        return len(self) > 0

    def __add__(self, other: Union['Document', Iterable['Document']]):
        raise NotImplementedError()
