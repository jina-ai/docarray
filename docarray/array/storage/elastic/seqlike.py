from typing import Union, Iterable

from ..base.seqlike import BaseSequenceLikeMixin
from .... import Document
from ..registry import _REGISTRY


class SequenceLikeMixin(BaseSequenceLikeMixin):
    """Implement sequence-like methods for DocumentArray with Elastic as storage"""

    def __eq__(self, other):
        """Compare this object to the other, returns True if and only if other
        as the same type as self and other has the same meta information

        :param other: the other object to check for equality
        :return: ``True`` if other is equal to self
        """
        # two DAW are considered as the same if they have the same client meta data
        return (
            type(self) is type(other)
            and self._client.get_meta() == other._client.get_meta()
            and self._config == other._config
        )

    def __len__(self):
        """Return the length of :class:`DocumentArray` that uses Elastic as storage

        :return: the length of this :class:`DocumentArrayElastic` object
        """
        return self._client.count(index=self._config.index_name)["count"]

    def _doc_id_exists(doc_id, client, elastic_config):
        return client.exists(index=elastic_config.index_name, id=doc_id)

    def __contains__(self, x: Union[str, 'Document']):
        """Check if ``x`` is contained in this :class:`DocumentArray` with Elastic storage

        :param x: the id of the document to check or the document object itself
        :return: True if ``x`` is contained in self
        """
        if isinstance(x, str):
            return self._doc_id_exists(x, self._client, self._config)
        elif isinstance(x, Document):
            return self._doc_id_exists(x.id, self._client, self._config)
        else:
            return False

    def __del__(self):
        """Delete this :class:`DocumentArrayElastic` object"""
        super().__del__()
        if (
            not self._persist
            and len(_REGISTRY[self.__class__.__name__][self._class_name]) == 1
        ):
            self._client.schema.delete_class(self._class_name)
            self._client.schema.delete_class(self._meta_name)
        _REGISTRY[self.__class__.__name__][self._class_name].remove(self)

    def __repr__(self):
        """Return the string representation of :class:`DocumentArrayElastic` object
        :return: string representation of this object
        """
        return f'<{self.__class__.__name__} (length={len(self)}) at {id(self)}>'

    def extend(self, values: Iterable['Document']) -> None:
        """Extends the array with the given values

        :param values: Documents to be added
        """

        request = []
        for value in values:
            value.embedding = self._map_embedding(value.embedding)
            request.append(
                {
                    "_op_type": "index",
                    '_id': value.id,
                    '_index': self._config.index_name,
                    'embedding': value.embedding,
                    'blob': value.to_base64(),
                }
            )
            self._offset2ids.append(value.id)

        if len(request) > 0:
            self._send_requests(request)
            self._refresh(self._config.index_name)