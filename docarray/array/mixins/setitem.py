import itertools
from typing import (
    TYPE_CHECKING,
    Union,
    Sequence,
    overload,
    Any,
    List,
)

import numpy as np

from ... import Document
from ...helper import typename

if TYPE_CHECKING:
    from ...types import (
        DocumentArrayIndexType,
        DocumentArraySingletonIndexType,
        DocumentArrayMultipleIndexType,
        DocumentArrayMultipleAttributeType,
        DocumentArraySingleAttributeType,
    )


class SetItemMixin:
    """Provides helper function to allow advanced indexing for `__setitem__`"""

    @overload
    def __setitem__(
        self,
        index: 'DocumentArrayMultipleAttributeType',
        value: List[List['Any']],
    ):
        ...

    @overload
    def __setitem__(
        self,
        index: 'DocumentArraySingleAttributeType',
        value: List['Any'],
    ):
        ...

    @overload
    def __setitem__(
        self,
        index: 'DocumentArraySingletonIndexType',
        value: 'Document',
    ):
        ...

    @overload
    def __setitem__(
        self,
        index: 'DocumentArrayMultipleIndexType',
        value: Sequence['Document'],
    ):
        ...

    def __setitem__(
        self,
        index: 'DocumentArrayIndexType',
        value: Union['Document', Sequence['Document']],
    ):

        if isinstance(index, (int, np.generic)) and not isinstance(index, bool):
            self._set_doc_by_offset(int(index), value)
        elif isinstance(index, str):
            if index.startswith('@'):
                self._set_doc_value_pairs(self.traverse_flat(index[1:]), value)
            else:
                self._set_doc_by_id(index, value)
        elif isinstance(index, slice):
            self._set_docs_by_slice(index, value)
        elif index is Ellipsis:
            self._set_doc_value_pairs(self.flatten(), value)
        elif isinstance(index, Sequence):
            if isinstance(index, tuple) and len(index) == 2:
                self._set_by_pair(index[0], index[1], value)

            elif isinstance(index[0], bool):
                self._set_by_mask(index[0], value)

            elif isinstance(index[0], (int, str)):
                # if single value
                if isinstance(value, str) or not isinstance(value, Sequence):
                    for si in index:
                        self[si] = value  # leverage existing setter
                else:
                    for si, _val in zip(index, value):
                        self[si] = _val  # leverage existing setter

        elif isinstance(index, np.ndarray):
            index = index.squeeze()
            if index.ndim == 1:
                self[index.tolist()] = value  # leverage existing setter
            else:
                raise IndexError(
                    f'When using np.ndarray as index, its `ndim` must =1. However, receiving ndim={index.ndim}'
                )
        else:
            raise IndexError(f'Unsupported index type {typename(index)}: {index}')

    def _set_by_pair(self, idx1, idx2, value):
        if isinstance(idx1, str):
            # second is an ID
            if isinstance(idx2, str) and idx2 in self:
                self._set_doc_value_pairs((self[idx1], self[idx2]), value)
            # second is an attribute
            elif isinstance(idx2, str) and hasattr(self[idx1], idx2):
                self._set_doc_attr_by_id(idx1, idx2, value)
            # second is a list of attributes:
            elif (
                isinstance(idx2, Sequence)
                and all(isinstance(attr, str) for attr in idx2)
                and all(hasattr(self[idx1], attr) for attr in idx2)
            ):
                for attr, _v in zip(idx2, value):
                    self._set_doc_attr_by_id(idx1, attr, _v)
            else:
                raise ValueError(f'`{idx2}` is neither a valid id nor attribute name')
        elif isinstance(idx1, int):
            # second is an offset:
            if isinstance(idx2, int):
                self._set_doc_value_pairs((self[idx1], self[idx2]), value)
            # second is an attribute
            elif isinstance(idx2, str) and hasattr(self[idx1], idx2):
                self._set_doc_attr_by_id(idx1, idx2, value)
            # second is a list of attributes:
            elif (
                isinstance(idx2, Sequence)
                and all(isinstance(attr, str) for attr in idx2)
                and all(hasattr(self[idx1], attr) for attr in idx2)
            ):
                for attr, _v in zip(idx2, value):
                    self._set_doc_attr_by_id(idx1, attr, _v)
            else:
                raise ValueError(f'`{idx2}` must be an attribute or list of attributes')

        elif isinstance(idx1, (slice, Sequence)) or idx1 is Ellipsis:
            self._set_docs_attributes(idx1, idx2, value)

    def _set_by_mask(self, mask: List[bool], value):
        if len(mask) != len(self):
            raise IndexError(
                f'Boolean mask index is required to have the same length as {len(self)}, '
                f'but receiving {len(mask)}'
            )
        _selected = itertools.compress(self, mask)
        self._set_doc_value_pairs(_selected, value)

    def _set_docs_attributes(self, index, attributes, value):
        # TODO: handle index is Ellipsis
        if isinstance(attributes, str):
            # a -> [a]
            # [a, a] -> [a, a]
            attributes = (attributes,)
        if isinstance(value, (list, tuple)) and not any(
            isinstance(el, (tuple, list)) for el in value
        ):
            # [x] -> [[x]]
            # [[x], [y]] -> [[x], [y]]
            value = (value,)
        if not isinstance(value, (list, tuple)):
            # x -> [x]
            value = (value,)

        _docs = self[index]
        if not _docs:
            return

        for _a, _v in zip(attributes, value):
            if _a in ('tensor', 'embedding'):
                if _a == 'tensor':
                    _docs.tensors = _v
                elif _a == 'embedding':
                    _docs.embeddings = _v
                for _d in _docs:
                    self._set_doc_by_id(_d.id, _d)
            else:
                if not isinstance(_v, (list, tuple)):
                    for _d in _docs:
                        self._set_doc_attr_by_id(_d.id, _a, _v)
                else:
                    for _d, _vv in zip(_docs, _v):
                        self._set_doc_attr_by_id(_d.id, _a, _vv)
