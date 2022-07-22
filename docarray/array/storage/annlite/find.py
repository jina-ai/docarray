from typing import (
    Union,
    Optional,
    TYPE_CHECKING,
    List,
    Dict,
)

if TYPE_CHECKING:
    import numpy as np

from .... import DocumentArray


class FindMixin:
    def _find(
        self,
        query: 'np.ndarray',
        limit: Optional[Union[int, float]] = 20,
        only_id: bool = False,
        filter: Optional[Dict] = None,
        secondary_index: Optional[str] = None,
        **kwargs,
    ) -> List['DocumentArray']:
        """Returns approximate nearest neighbors given an input query.

        :param query: the query documents to search.
        :param limit: the number of results to get for each query document in search.
        :param only_id: if set, then returning matches will only contain ``id``
        :param filter: filter query used for pre-filtering
        :param secondary_index: if set, then the returned DocumentArray will be retrieved from the given secondary index.
        :param kwargs: other kwargs.

        :return: a list of DocumentArrays containing the closest Document objects for each of the queries in `query`.
        """
        from ....math import ndarray

        if secondary_index in self._secondary_indices.keys():
            _annlite = self._secondary_indices[secondary_index]._annlite
        else:
            _annlite = self._annlite

        n_rows, _ = ndarray.get_array_rows(query)
        if n_rows == 1:
            query = query.reshape(1, -1)

        _, match_docs = _annlite._search_documents(
            query, limit=limit, filter=filter or {}, include_metadata=not only_id
        )

        return match_docs

    def _filter(
        self,
        filter: Dict,
        limit: Optional[Union[int, float]] = 20,
        only_id: bool = False,
        secondary_index: Optional[str] = None,
    ) -> 'DocumentArray':
        """Returns a subset of documents by filtering by the given filter (`Annlite` filter).

        :param filter: the input filter to apply in each stored document
        :param limit: the number of results to get for each query document in search.
        :param only_id: if set, then returning matches will only contain ``id``
        :param secondary_index: if set, then the returned DocumentArray will be retrieved from the given secondary index.
        :return: a `DocumentArray` containing the `Document` objects that verify the filter.
        """

        if secondary_index in self._secondary_indices.keys():
            _annlite = self._secondary_indices[secondary_index]._annlite
        else:
            _annlite = self._annlite

        docs = _annlite.filter(
            filter=filter,
            limit=limit,
            include_metadata=not only_id,
        )
        return DocumentArray(docs)
