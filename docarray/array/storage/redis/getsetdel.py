from typing import Iterable, Dict

from ..base.getsetdel import BaseGetSetDelMixin
from ..base.helper import Offset2ID
from .... import Document


class GetSetDelMixin(BaseGetSetDelMixin):
    """Provide concrete implementation for ``__getitem__``, ``__setitem__``,
    and ``__delitem__`` for ``DocumentArrayRedis``"""

    def _getitem(self, doc_id: str) -> 'Document':
        """Helper method for getting item with weaviate as storage

        :param doc_id:  id of the document
        :raises KeyError: raise error when redis id does not exist in storage
        :return: Document
        """
        try:
            result = self._client.get(self._config.key_prefix + doc_id)
            # doc = Document.from_base64(result['_source']['blob'])
            # return doc

        except Exception as ex:
            raise KeyError(doc_id) from ex

    def _get_doc_by_id(self, _id: str) -> 'Document':
        """Concrete implementation of base class' ``_get_doc_by_id``

        :param _id: the id of the document
        :return: the retrieved document from redis
        """
        return self._getitem(_id)

    def _set_doc_by_id(self, _id: str, value: 'Document', flush: bool = True):
        """Concrete implementation of base class' ``_set_doc_by_id``

        :param _id: the id of doc to update
        :param value: the document to update to
        """
        if _id != value.id:
            self._del_doc_by_id(_id)

        payload = self._document_to_redis(value)
        self._add(_id, payload)

    def _del_doc_by_id(self, _id: str):
        """Concrete implementation of base class' ``_del_doc_by_id``

        :param _id: the id of the document to delete
        """
        if self._doc_id_exists(self._config.key_prefix + _id):
            self._client.delete(self._config.key_prefix + _id)

    def _document_to_redis(self, doc: 'Document') -> Dict:
        extra_columns = {col: doc.tags.get(col) for col, _ in self._config.columns}
        payload = {
            'embedding': self._map_embedding(doc.embedding),
            'blob': doc.to_base64(),
            **extra_columns,
        }

        if self._config.tag_indices:
            for index in self._config.tag_indices:
                payload[index] = doc.tags.get(index)

        if doc.text:
            payload['text'] = doc.text
        return payload

    def _load_offset2ids(self):
        ids = self._get_offset2ids_meta()
        self._offset2ids = Offset2ID(ids)

    def _save_offset2ids(self):
        self._update_offset2ids_meta()
