from typing import Union, Iterable

from ..base.seqlike import BaseSequenceLikeMixin
from ...memory import DocumentArrayInMemory
from .... import Document


class SequenceLikeMixin(BaseSequenceLikeMixin):
    """Implement sequence-like methods"""

    def extend(self, values: Iterable['Document']) -> None:
        docs = DocumentArrayInMemory(values)
        if len(docs) == 0:
            return

        for doc in docs:
            doc.embedding = self._map_embedding(doc.embedding)

        self._annlite.index(docs)
        self._offset2ids.extend([doc.id for doc in docs])

        if self._secondary_indices:
            for selector, da in self._secondary_indices.items():
                docs_selector = docs[selector]
                if len(docs_selector) > 0:
                    da.extend(docs_selector)

    def append(self, value: 'Document'):
        self.extend([value])

    def __del__(self) -> None:
        if not self._persist:
            self._offset2ids.clear()
            self._annlite.clear()

    def __eq__(self, other):
        """In annlite backend, data are considered as identical if configs point to the same database source"""
        return (
            type(self) is type(other)
            and type(self._config) is type(other._config)
            and self._config == other._config
        )

    def __repr__(self):
        return f'<DocumentArray[AnnLite] (length={len(self)}) at {id(self)}>'

    def __contains__(self, x: Union[str, 'Document']):
        if isinstance(x, str):
            return self._annlite.get_doc_by_id(x) is not None
        elif isinstance(x, Document):
            return self._annlite.get_doc_by_id(x.id) is not None
        else:
            return False
