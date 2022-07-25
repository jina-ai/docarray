from typing import Union, Optional, Iterable

from ..base.seqlike import BaseSequenceLikeMixin
from .... import Document


class SequenceLikeMixin(BaseSequenceLikeMixin):
    """Implement sequence-like methods"""

    def _insert_doc_at_idx(self, doc, idx: Optional[int] = None):
        ch_doc = self._document_to_ch(doc)
        if idx is None:
            idx = len(self)
        self._client.execute(
            f"""
                INSERT INTO {self._table_name}
                VALUES
                    ('{ch_doc['doc_id']}',
                     '{ch_doc['serialized_value']}',
                      {ch_doc['embedding']},
                     '{ch_doc['text']}',
                     '{idx}', '{idx}')
            """
        )
        self._offset2ids.insert(idx, doc.id)

    def _shift_index_right_backward(self, start: int):
        idx = len(self) - 1
        while idx >= start:
            self._client.execute(
                f"""
                    ALTER TABLE {self._table_name}
                    UPDATE item_order = {idx+1}
                    WHERE item_order = {idx}
                """
            )
            idx -= 1

    def insert(self, index: int, value: 'Document'):
        """Insert `doc` at `index`.

        :param index: Position of the insertion.
        :param value: The doc needs to be inserted.
        """
        length = len(self)
        if index < 0:
            index = length + index
        index = max(0, min(length, index))
        self._shift_index_right_backward(index)
        self._insert_doc_at_idx(doc=value, idx=index)

    def append(self, doc: 'Document') -> None:
        ch_doc = self._document_to_ch(doc)
        self._client.execute(
            f"""
                INSERT INTO {self._table_name}
                VALUES
                    ('{ch_doc['doc_id']}',
                    '{ch_doc['serialized_value']}',
                     {ch_doc['embedding']},
                    '{ch_doc['text']}',
                    '{len(self)}', '{len(self)}')
            """
        )
        self._offset2ids.append(doc.id)

    def __del__(self) -> None:
        super().__del__()
        if not self._persist:
            self._client.execute(
                f"""
                    ALTER TABLE metadata
                    DELETE
                    WHERE table_name={self._table_name}
                    AND container_type={self.__class__.__name__}
                """
            )
            self._client.execute(f'DROP TABLE IF EXISTS {self._table_name}')

    def __contains__(self, item: Union[str, 'Document']):
        if isinstance(item, str):
            self._client.execute(
                f"SELECT 1 FROM {self._table_name} WHERE startsWith(doc_id, '{item}') = 1"
            )
            r = self._client.fetchall()
            return len(list(r)) > 0
        elif isinstance(item, Document):
            return item.id in self  # fall back to str check
        else:
            return False

    def __len__(self) -> int:
        # for _ in range(10):
        request = self._client.execute(f'SELECT COUNT(*) FROM {self._table_name}')
        request = self._client.fetchone()
        return request[0]

    def __repr__(self):
        return f'<DocumentArray[ClickHouse] (length={len(self)}) at {id(self)}>'

    def __eq__(self, other):
        """In ClockHouse backend, data are considered as identical if configs point to the same database source"""
        return (
            type(self) is type(other)
            and type(self._config) is type(other._config)
            and self._config == other._config
        )

    def extend(self, docs: Iterable['Document']) -> None:

        self_len = len(self)
        ch_docs = []
        for doc in docs:
            ch_doc = self._document_to_ch(doc)
            ch_doc['item_order'] = self_len
            ch_doc['indx'] = self_len
            ch_docs.append(ch_doc)

            self._offset2ids.append(doc.id)
            self_len += 1

        self._client.execute(f"INSERT INTO {self._table_name} VALUES", ch_docs)
