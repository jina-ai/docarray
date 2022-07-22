from typing import Iterable, Dict

from ..base.getsetdel import BaseGetSetDelMixin
from ..base.helper import Offset2ID
from .... import Document


class GetSetDelMixin(BaseGetSetDelMixin):
    """Provide concrete implementation for ``__getitem__``, ``__setitem__``,
    and ``__delitem__`` for ``DocumentArrayElastic``"""

    def _document_to_elastic(self, doc: 'Document') -> Dict:
        extra_columns = {col: doc.tags.get(col) for col, _ in self._config.columns}
        request = {
            '_op_type': 'index',
            '_id': doc.id,
            '_index': self._config.index_name,
            'embedding': self._map_embedding(doc.embedding),
            'blob': doc.to_base64(),
            **extra_columns,
        }

        if self._config.tag_indices:
            for index in self._config.tag_indices:
                request[index] = doc.tags.get(index)

        if doc.text:
            request['text'] = doc.text
        return request

    def _getitem(self, doc_id: str) -> 'Document':
        """Helper method for getting item with elastic as storage

        :param doc_id:  id of the document
        :raises KeyError: raise error when elastic id does not exist in storage
        :return: Document
        """
        try:
            result = self._client.get(index=self._config.index_name, id=doc_id)
            doc = Document.from_base64(result['_source']['blob'])
            return doc
        except Exception as ex:
            raise KeyError(doc_id) from ex

    def _get_doc_by_id(self, _id: str) -> 'Document':
        """Concrete implementation of base class' ``_get_doc_by_id``

        :param _id: the id of the document
        :return: the retrieved document from elastic
        """
        return self._getitem(_id)

    def _set_doc_by_id(self, _id: str, value: 'Document'):
        """Concrete implementation of base class' ``_set_doc_by_id``

        :param _id: the id of doc to update
        :param value: the document to update to
        """
        if _id != value.id:
            self._del_doc_by_id(_id)

        request = [self._document_to_elastic(value)]

        self._send_requests(request)
        self._refresh(self._config.index_name)

    def _set_docs_by_ids(self, ids, docs: Iterable['Document'], mismatch_ids: Dict):
        """Overridden implementation of _set_docs_by_ids in order to add docs in batches and flush at the end

        :param ids: the ids used for indexing
        """
        for _id, doc in zip(ids, docs):
            self._set_doc_by_id(_id, doc)

        self._refresh(self._config.index_name)

    def _del_doc_by_id(self, _id: str):
        """Concrete implementation of base class' ``_del_doc_by_id``

        :param _id: the id of the document to delete
        """
        if self._doc_id_exists(_id):
            self._client.delete(index=self._config.index_name, id=_id)

        self._refresh(self._config.index_name)

    def _clear_storage(self):
        """Concrete implementation of base class' ``_clear_storage``"""
        self._client.indices.delete(index=self._config.index_name)

    def _load_offset2ids(self):
        ids = self._get_offset2ids_meta()
        self._offset2ids = Offset2ID(ids)

    def _save_offset2ids(self):
        self._update_offset2ids_meta()
