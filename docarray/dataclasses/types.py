import base64
import copy
import functools
import typing
from dataclasses import (
    dataclass as _dataclass,
    Field as _Field,
    is_dataclass as _is_dataclass,
    field as _field,
    MISSING,
)
from enum import Enum
from typing import (
    TypeVar,
    Callable,
    Optional,
    overload,
    Dict,
    Type,
)

from .getter import *
from .setter import *

from functools import partial

import sys

if sys.version_info >= (3, 9):
    from typing import get_type_hints, get_args
else:
    from typing_extensions import get_type_hints, get_args

if TYPE_CHECKING:
    import numpy as np
    from ..typing import T
    from docarray import Document

from ..typing import Image, Text, Audio, Video, Mesh, Tabular, Blob, JSON

__all__ = ['field', 'dataclass', 'is_multimodal']


class AttributeType(str, Enum):
    DOCUMENT = 'document'
    PRIMITIVE = 'primitive'
    ITERABLE_PRIMITIVE = 'iterable_primitive'
    ITERABLE_DOCUMENT = 'iterable_document'
    NESTED = 'nested'
    ITERABLE_NESTED = 'iterable_nested'


class Field(_Field):
    def __init__(
        self,
        *,
        setter: Optional[Callable] = None,
        getter: Optional[Callable] = None,
        _source_field: Optional[_Field] = None,
        **kwargs,
    ):
        self.copy_from(_source_field if _source_field else _field(**kwargs))
        self.setter = setter
        self.getter = getter

    def copy_from(self, f: '_Field'):
        for s in f.__slots__:
            setattr(self, s, getattr(f, s))


@overload
def field(
    *,
    _source_field: Optional[_Field] = None,  # Privately used
    setter: Optional[Callable] = None,
    getter: Optional[Callable] = None,
    default=MISSING,
    default_factory=MISSING,
    init=True,
    repr=True,
    hash=None,
    compare=True,
    metadata=None,
) -> Field:
    ...


def field(**kwargs) -> Field:
    """
    Creates new multimodal type for a DocArray dataclass.

    :meth:`field` is used to define the *get* and *set* behaviour of custom types when used in a DocArray dataclass.

    .. code-block:: python

        from docarray import Document, dataclass, field
        from typing import TypeVar

        MyImage = TypeVar('MyImage', bound=str)


        def my_setter(value) -> 'Document':
            return Document(uri=value).load_uri_to_blob()


        def my_getter(doc: 'Document'):
            return doc.uri


        @dataclass
        class MMDoc:
            banner: MyImage = field(setter=my_setter, getter=my_getter, default='test-1.jpeg')

    """
    return Field(**kwargs)


def _is_field(f) -> bool:
    return isinstance(f, Field) and getattr(f, 'setter') and getattr(f, 'getter')


_TYPES_REGISTRY = {
    Image: lambda x: field(setter=image_setter, getter=image_getter, _source_field=x),
    'Image_': lambda height, width, axis, x: field(
        setter=partial(image_setter_, height, width, axis),
        getter=image_getter,
        _source_field=x,
    ),
    Text: lambda x: field(setter=text_setter, getter=text_getter, _source_field=x),
    Audio: lambda x: field(setter=audio_setter, getter=audio_getter, _source_field=x),
    JSON: lambda x: field(setter=json_setter, getter=json_getter, _source_field=x),
    Video: lambda x: field(setter=video_setter, getter=video_getter, _source_field=x),
    Tabular: lambda x: field(
        setter=tabular_setter, getter=tabular_getter, _source_field=x
    ),
    Blob: lambda x: field(setter=blob_setter, getter=blob_getter, _source_field=x),
    Mesh: lambda x: field(setter=mesh_setter, getter=mesh_getter, _source_field=x),
}


@overload
def dataclass(
    cls: Optional['T'] = None,
    *,
    init: bool = True,
    repr: bool = True,
    eq: bool = True,
    order: bool = False,
    unsafe_hash: bool = False,
    frozen: bool = False,
    type_var_map: Optional[Dict[TypeVar, Callable[['_Field'], 'Field']]] = None,
) -> 'T':
    ...


def dataclass(
    cls: Optional['T'] = None,
    *,
    type_var_map: Optional[Dict[TypeVar, Callable[['_Field'], 'Field']]] = None,
    **kwargs,
) -> 'T':
    """Annotates a class as a DocArray dataclass type.

    Example usage:

    >>> from docarray.typing import Image, Text
    >>> from docarray import dataclass
    >>> @dataclass:
    >>> class X:
    >>>     banner: Image = 'apple.png'
    >>>     description: Text = 'This is a big red apple.'

    :param type_var_map: a mapping from TypeVar to a callable that gives Field.

        .. highlight:: python
        .. code-block:: python

            _TYPES_REGISTRY = {
                Image: lambda x: field(setter=image_setter, getter=image_getter, _source_field=x),
                Text: lambda x: field(setter=text_setter, getter=text_getter, _source_field=x),
            }

        The default mapping will be overrided by this new mapping if they collide on the keys.

    """

    if not type_var_map:
        type_var_map = _TYPES_REGISTRY
    else:
        r = copy.deepcopy(_TYPES_REGISTRY)
        r.update(type_var_map)
        type_var_map = r

    from docarray import Document

    def deco(f):
        """
        Set Decorator function.

        :param f: function the decorator is used for
        :return: wrapper
        """

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            if not kwargs and len(args) == 2 and isinstance(args[1], Document):
                return f(args[0], **_from_document(type(args[0]), args[1]))
            else:
                return f(*args, **kwargs)

        return wrapper

    def wrap(cls):
        decorated_cls = _dataclass(cls, **kwargs)

        # inject flag for recognizing this is a multimodal dataclass
        setattr(decorated_cls, '__is_multimodal__', True)

        # wrap init so `MMDoc(document)` is possible
        if getattr(decorated_cls, '__init__'):
            decorated_cls.__init__ = deco(decorated_cls.__init__)

        type_hints = get_type_hints(cls, include_extras=True)
        for name, type_hint in type_hints.items():
            type_args = get_args(type_hint)
            if type_args:
                main_type, extras = type_args[0], type_args[1:]
                if main_type == Image:
                    f = decorated_cls.__dataclass_fields__[name]
                    decorated_cls.__dataclass_fields__[name] = type_var_map['Image_'](
                        *extras, f
                    )

        for key, f in decorated_cls.__dataclass_fields__.items():
            if _is_field(f):
                continue

            if f.type in type_var_map:
                decorated_cls.__dataclass_fields__[key] = type_var_map[f.type](f)

            elif isinstance(f.type, typing._GenericAlias) and f.type._name in [
                'List',
                'Iterable',
            ]:
                sub_type = f.type.__args__[0]
                if sub_type in type_var_map:
                    decorated_cls.__dataclass_fields__[key] = type_var_map[sub_type](f)

        return decorated_cls

    if cls is None:
        return wrap

    return wrap(cls)


def is_multimodal(obj) -> bool:
    """Returns True if obj is an instance of :meth:`dataclass`."""
    from docarray import Document

    if isinstance(obj, Document):
        return obj.is_multimodal
    else:
        return _is_dataclass(obj) and hasattr(obj, '__is_multimodal__')


def _from_document(cls: Type['T'], doc: 'Document') -> 'T':
    if not doc.is_multimodal:
        raise ValueError(
            f'{doc} is not a multimodal doc instantiated from a class wrapped by `docarray.dataclasses.tdataclass`.'
        )

    attributes = {}
    for key, attribute_info in doc._metadata['multi_modal_schema'].items():
        field = cls.__dataclass_fields__[key]
        position = doc._metadata['multi_modal_schema'][key].get('position')
        if (
            attribute_info['type'] == 'bytes'
            and attribute_info['attribute_type'] == AttributeType.PRIMITIVE
        ):
            attributes[key] = base64.b64decode(doc.tags[key].encode())
        elif attribute_info['attribute_type'] in [
            AttributeType.PRIMITIVE,
            AttributeType.ITERABLE_PRIMITIVE,
        ]:
            attributes[key] = doc.tags[key]
        elif attribute_info['attribute_type'] == AttributeType.DOCUMENT:
            attribute_doc = doc.chunks[int(position)]
            attribute = _get_doc_attribute(attribute_doc, field)
            attributes[key] = attribute
        elif attribute_info['attribute_type'] == AttributeType.ITERABLE_DOCUMENT:
            attribute_list = []
            for chunk_doc in doc.chunks[int(position)].chunks:
                attribute_list.append(_get_doc_attribute(chunk_doc, field))
            attributes[key] = attribute_list
        elif attribute_info['attribute_type'] == AttributeType.NESTED:
            nested_cls = field.type
            attributes[key] = _get_doc_nested_attribute(
                doc.chunks[int(position)], nested_cls
            )
        elif attribute_info['attribute_type'] == AttributeType.ITERABLE_NESTED:
            nested_cls = cls.__dataclass_fields__[key].type.__args__[0]
            attribute_list = []
            for chunk_doc in doc.chunks[int(position)].chunks:
                attribute_list.append(_get_doc_nested_attribute(chunk_doc, nested_cls))
            attributes[key] = attribute_list
        else:
            raise AttributeError(f'Invalid attribute at `{key}`')

    return attributes


def _get_doc_attribute(attribute_doc: 'Document', field):
    if _is_field(field):
        return field.getter(attribute_doc)
    else:
        raise ValueError('Invalid attribute type')


def _get_doc_nested_attribute(attribute_doc: 'Document', nested_cls: Type['T']) -> 'T':
    if not is_multimodal(nested_cls):
        raise ValueError(f'Nested attribute {nested_cls.__name__} is not a dataclass')
    return nested_cls(**_from_document(nested_cls, attribute_doc))
