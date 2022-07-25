from operator import itemgetter
from typing import Dict

from ..base.getsetdel import BaseGetSetDelMixin
from ..base.helper import Offset2ID
from .... import Document


class GetSetDelMixin(BaseGetSetDelMixin):
    """Implement required and derived functions that power `getitem`, `setitem`, `delitem`"""

    def _document_to_ch(self, doc: 'Document') -> Dict:
        ch_doc = {
            'doc_id': doc.id,
            'serialized_value': doc.to_base64(),
            'embedding': self._map_embedding(doc.embedding),
            'text': '',
        }

        if doc.text:
            ch_doc['text'] = doc.text
        return ch_doc

    # essential methods start

    def _del_doc_by_id(self, _id: str):
        self._client.execute(
            f"ALTER TABLE {self._table_name} DELETE WHERE startsWith(doc_id, '{_id}') = 1"
        )
        self._offset2ids.delete_by_id(_id)

    def _set_doc_by_id(self, _id: str, value: 'Document'):
        ch_doc = self._document_to_ch(value)

        self._client.execute(
            f"""
                    ALTER TABLE {self._table_name}
                    UPDATE doc_id= '{ch_doc['doc_id']}',
                        serialized_value = '{ch_doc['serialized_value']}',
                        embedding={ch_doc['embedding']},
                        text='{ch_doc['text']}'
                    WHERE startsWith(doc_id, '{_id}') = 1
                """
        )

    def _get_doc_by_id(self, id: str) -> 'Document':
        self._client.execute(
            f"SELECT serialized_value FROM {self._table_name} WHERE startsWith(doc_id, '{id}') = 1"
        )
        res = self._client.fetchone()
        if res is None:
            raise KeyError(f'Can not find Document with id=`{id}`')
        doc = Document.from_base64(res[0])
        return doc

    def _load_offset2ids(self):
        self._client.execute(
            f"SELECT doc_id FROM {self._table_name} ORDER BY item_order",
        )
        ids = self._client.fetchall()
        self._offset2ids = Offset2ID(list(map(itemgetter(0), ids)))

    def _save_offset2ids(self):
        for offset, doc_id in enumerate(self._offset2ids):
            self._client.execute(
                f"ALTER TABLE {self._table_name} UPDATE item_order = {offset} WHERE doc_id = '{doc_id}'"
            )

    # essentials end here

    def _clear_storage(self):
        self._client.execute(f'TRUNCATE TABLE {self._table_name}')
