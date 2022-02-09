from typing import List, Iterable, Iterator, TYPE_CHECKING

import scipy.sparse
from qdrant_client import QdrantClient
from qdrant_openapi_client.exceptions import UnexpectedResponse
from qdrant_openapi_client.models.models import PointIdsList, PointsList, ScrollRequest

from docarray import Document
from docarray.array.storage.base.getsetdel import BaseGetSetDelMixin
from docarray.array.storage.qdrant.helper import QdrantStorageHelper


class GetSetDelMixin(BaseGetSetDelMixin):
    @property
    def client(self) -> QdrantClient:
        raise NotImplementedError()

    @property
    def serialization_config(self) -> dict:
        raise NotImplementedError()

    @property
    def n_dim(self) -> int:
        raise NotImplementedError()

    @property
    def collection_name(self) -> str:
        raise NotImplementedError()

    @property
    def scroll_batch_size(self) -> int:
        raise NotImplementedError()

    def _upload_batch(self, docs: Iterable[Document]):
        self.client.http.points_api.upsert_points(
            name=self.collection_name,
            wait=True,
            point_insert_operations=PointsList(
                points=[self._document_to_qdrant(doc) for doc in docs]
            ),
        )

    def _qdrant_to_document(self, qdrant_record: dict) -> 'Document':
        return Document.from_base64(
            qdrant_record['payload']['_serialized'], **self.serialization_config
        )

    def _document_to_qdrant(self, doc: Document) -> dict:
        return dict(
            id=doc.id,
            payload=dict(_serialized=doc.to_base64(**self.serialization_config)),
            vector=QdrantStorageHelper.embedding_to_array(doc.embedding, self.n_dim),
        )

    def _get_doc_by_offset(self, offset: int) -> 'Document':
        raise NotImplementedError()

    def _get_doc_by_id(self, _id: str) -> 'Document':
        try:
            resp = self.client.http.points_api.get_point(
                name=self.collection_name, id=_id
            )
            return self._qdrant_to_document(resp.json())
        except UnexpectedResponse as response_error:
            if response_error.status_code == 404:
                raise KeyError(_id)

    def _del_doc_by_offset(self, offset: int):
        raise NotImplementedError()

    def _del_doc_by_id(self, _id: str):
        self.client.http.points_api.delete_points(
            name=self.collection_name,
            wait=True,
            points_selector=PointIdsList(points=[_id]),
        )

    def _set_doc_by_offset(self, offset: int, value: 'Document'):
        raise NotImplementedError()

    def _set_doc_by_id(self, _id: str, value: 'Document'):
        if _id != value.id:
            self._del_doc_by_id(_id)
        self.client.http.points_api.upsert_points(
            name=self.collection_name,
            wait=True,
            point_insert_operations=[self._document_to_qdrant(value)],
        )

    def scan(self) -> Iterator['Document']:
        offset = None
        while True:
            response = self.client.http.points_api.scroll_points(
                name=self.collection_name,
                scroll_request=ScrollRequest(
                    offset=offset,
                    limit=self.scroll_batch_size,
                    with_payload=['_serialized'],
                    with_vector=False,
                ),
            )
            for point in response.result.points:
                yield self._qdrant_to_document(point.json())

            if response.result.next_page_offset:
                offset = response.result.next_page_offset
            else:
                break