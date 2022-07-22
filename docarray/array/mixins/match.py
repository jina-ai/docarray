from typing import Optional, Union, Callable, Tuple, TYPE_CHECKING, Dict

if TYPE_CHECKING:
    import numpy as np
    from ...typing import ArrayType
    from ... import DocumentArray


class MatchMixin:
    """A mixin that provides match functionality to DocumentArrays"""

    def match(
        self,
        darray: 'DocumentArray',
        metric: Union[
            str, Callable[['ArrayType', 'ArrayType'], 'np.ndarray']
        ] = 'cosine',
        limit: Optional[Union[int, float]] = 20,
        normalization: Optional[Tuple[float, float]] = None,
        metric_name: Optional[str] = None,
        batch_size: Optional[int] = None,
        exclude_self: bool = False,
        filter: Optional[Dict] = None,
        only_id: bool = False,
        use_scipy: bool = False,
        device: str = 'cpu',
        num_worker: Optional[int] = 1,
        secondary_index: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Compute embedding based nearest neighbour in `another` for each Document in `self`,
        and store results in `matches`.
        .. note::
            'cosine', 'euclidean', 'sqeuclidean' are supported natively without extra dependency.
            You can use other distance metric provided by ``scipy``, such as `braycurtis`, `canberra`, `chebyshev`,
            `cityblock`, `correlation`, `cosine`, `dice`, `euclidean`, `hamming`, `jaccard`, `jensenshannon`,
            `kulsinski`, `mahalanobis`, `matching`, `minkowski`, `rogerstanimoto`, `russellrao`, `seuclidean`,
            `sokalmichener`, `sokalsneath`, `sqeuclidean`, `wminkowski`, `yule`.
            To use scipy metric, please set ``use_scipy=True``.
        - To make all matches values in [0, 1], use ``dA.match(dB, normalization=(0, 1))``
        - To invert the distance as score and make all values in range [0, 1],
            use ``dA.match(dB, normalization=(1, 0))``. Note, how ``normalization`` differs from the previous.
        - If a custom metric distance is provided. Make sure that it returns scores as distances and not similarity, meaning the smaller the better.
        :param darray: the other DocumentArray  to match against
        :param metric: the distance metric
        :param limit: the maximum number of matches, when not given defaults to 20.
        :param normalization: a tuple [a, b] to be used with min-max normalization,
                                the min distance will be rescaled to `a`, the max distance will be rescaled to `b`
                                all values will be rescaled into range `[a, b]`.
        :param metric_name: if provided, then match result will be marked with this string.
        :param batch_size: if provided, then ``darray`` is loaded in batches, where each of them is at most ``batch_size``
            elements. When `darray` is big, this can significantly speedup the computation.
        :param exclude_self: if set, Documents in ``darray`` with same ``id`` as the left-hand values will not be
                        considered as matches.
        :param filter: filter query used for pre-filtering
        :param only_id: if set, then returning matches will only contain ``id``
        :param use_scipy: if set, use ``scipy`` as the computation backend. Note, ``scipy`` does not support distance
            on sparse matrix.
        :param device: the computational device for ``.match()``, can be either `cpu` or `cuda`.
        :param num_worker: the number of parallel workers. If not given, then the number of CPUs in the system will be used.

                .. note::
                    This argument is only effective when ``batch_size`` is set.
        :param kwargs: other kwargs.
        """

        if not (self and darray):
            return

        for d in self:
            d.matches.clear()

        match_docs = darray.find(
            self,
            metric=metric,
            limit=limit,
            normalization=normalization,
            metric_name=metric_name,
            batch_size=batch_size,
            exclude_self=exclude_self,
            filter=filter,
            only_id=only_id,
            use_scipy=use_scipy,
            device=device,
            num_worker=num_worker,
        )

        if not isinstance(match_docs, list):
            match_docs = [match_docs]

        for m, d in zip(match_docs, self):
            d.matches = m
