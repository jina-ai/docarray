from typing import (
    TYPE_CHECKING,
    TypeVar,
    Sequence,
    List,
    Optional,
    Dict,
)

import numpy as np

from .... import Document, DocumentArray
from ....math import ndarray
from ....math.helper import EPSILON
from ....score import NamedScore
from ....array.mixins.find import FindMixin as BaseFindMixin

if TYPE_CHECKING:
    import tensorflow
    import torch

    CLickHouseArrayType = TypeVar(
        'CLickHouseArrayType',
        np.ndarray,
        tensorflow.Tensor,
        torch.Tensor,
        Sequence[float],
    )


class FindMixin(BaseFindMixin):
    def _find_similar_vectors(self, query: 'CLickHouseArrayType', limit=10):
        is_all_zero = np.all(query == 0)
        if is_all_zero:
            query = query + EPSILON

        query = query.tolist()

        resp = self._fetchall(
            f"""
                SELECT
                    serialized_value,
                    embedding,
                    L2Distance({query}, embedding)  AS dist 
                FROM {self._table_name}
                ORDER BY dist ASC LIMIT {limit}
            """
        )

        da = DocumentArray()
        for result in resp:
            doc = Document.from_base64(result[0])
            doc.embedding = result[1]
            doc.scores['score'] = NamedScore(value=result[2])
            da.append(doc)

        return da

    def _find(
        self,
        query: 'CLickHouseArrayType',
        limit: int = 10,
        filter: Optional[Dict] = None,
        **kwargs,
    ) -> List['DocumentArray']:
        """Returns approximate nearest neighbors given a batch of input queries.

        :param query: input supported to be stored in ClickHouse. This includes any from the list '[np.ndarray, tensorflow.Tensor, torch.Tensor, Sequence[float]]'
        :param limit: number of retrieved items
        :param filter: filter query used for pre-filtering

        :return: DocumentArray containing the closest documents to the query if it is a single query, otherwise a list of DocumentArrays containing
           the closest Document objects for each of the queries in `query`.
        """
        if filter is not None:
            raise ValueError(
                'Filtered vector search is not supported for ClickHouse backend'
            )
        query = np.array(query)
        num_rows, n_dim = ndarray.get_array_rows(query)
        if n_dim != 2:
            query = query.reshape((num_rows, -1))

        return [self._find_similar_vectors(q, limit=limit) for q in query]
