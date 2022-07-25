import base64
import typing

from ...dataclasses.types import Field, is_multimodal, _is_field
from ...dataclasses.types import AttributeType

if typing.TYPE_CHECKING:
    from docarray import Document, DocumentArray


class MultiModalMixin:
    @property
    def is_multimodal(self) -> bool:
        """
        Return true if this Document can be represented by a class wrapped
        by :meth:`docarray.dataclasses.types.dataclass`.
        """
        return 'multi_modal_schema' in self._metadata

    @classmethod
    def _from_dataclass(cls, obj) -> 'Document':
        if not is_multimodal(obj):
            raise TypeError(
                f'Object {type(obj).__name__} is not a `docarray.dataclass` instance'
            )

        from docarray import Document

        root = Document()
        tags = {}
        multi_modal_schema = {}
        for key, field in obj.__dataclass_fields__.items():
            attribute = getattr(obj, key)
            if attribute is None:
                continue

            if field.type in [str, int, float, bool] and not _is_field(field):
                tags[key] = attribute
                multi_modal_schema[key] = {
                    'attribute_type': AttributeType.PRIMITIVE,
                    'type': field.type.__name__,
                }

            elif field.type == bytes and not _is_field(field):
                tags[key] = base64.b64encode(attribute).decode()
                multi_modal_schema[key] = {
                    'attribute_type': AttributeType.PRIMITIVE,
                    'type': field.type.__name__,
                }
            elif isinstance(field.type, typing._GenericAlias):
                if field.type._name in ['List', 'Iterable']:
                    sub_type = field.type.__args__[0]
                    if sub_type in [str, int, float, bool]:
                        tags[key] = attribute
                        multi_modal_schema[key] = {
                            'attribute_type': AttributeType.ITERABLE_PRIMITIVE,
                            'type': f'{field.type._name}[{sub_type.__name__}]',
                        }

                    else:
                        chunk = Document()
                        for element in attribute:
                            doc, attribute_type = cls._from_obj(
                                element, sub_type, field
                            )
                            if attribute_type == AttributeType.DOCUMENT:
                                attribute_type = AttributeType.ITERABLE_DOCUMENT
                            elif attribute_type == AttributeType.NESTED:
                                attribute_type = AttributeType.ITERABLE_NESTED
                            else:
                                raise TypeError(
                                    f'Unsupported type annotation inside Iterable: {sub_type}'
                                )
                            chunk.chunks.append(doc)
                        multi_modal_schema[key] = {
                            'attribute_type': attribute_type,
                            'type': f'{field.type._name}[{sub_type.__name__}]',
                            'position': len(root.chunks),
                        }
                        root.chunks.append(chunk)
                else:
                    # this is obviously not generally correct logic here
                    # and also f*cking ugly, equally as obviuosly
                    # so this is just to make the POC work
                    import sys

                    if sys.version_info >= (3, 9):
                        from typing import get_type_hints, get_args
                    else:
                        from typing_extensions import get_type_hints, get_args

                    main_type = get_args(field.type)[0]
                    doc, attribute_type = cls._from_obj(attribute, field.type, field)
                    multi_modal_schema[key] = {
                        'attribute_type': attribute_type,
                        'type': main_type.__name__,
                        'position': len(root.chunks),
                    }
                    root.chunks.append(doc)
                # else:
                #     raise TypeError(
                #         f'Unsupported type annotation on field `{field.type._name}`'
                #     )
            else:
                doc, attribute_type = cls._from_obj(attribute, field.type, field)
                multi_modal_schema[key] = {
                    'attribute_type': attribute_type,
                    'type': field.type.__name__,
                    'position': len(root.chunks),
                }
                root.chunks.append(doc)

        # TODO: may have to modify this?
        root.tags = tags
        root._metadata['multi_modal_schema'] = multi_modal_schema

        return root

    def _get_mm_attr_postion(self, attr):
        if not self.is_multimodal:
            raise ValueError(
                'the Document does not correspond to a Multi Modal Document'
            )

        if attr not in self._metadata['multi_modal_schema']:
            raise ValueError(f'the Document schema does not contain attribute {attr}')

        pos = self._metadata['multi_modal_schema'][attr].get('position')
        if pos is None:
            raise ValueError(
                f'attribute {attr} is not a valid multi modal attribute.'
                f' One possible cause is the usage of a non-supported type in the dataclass definition.'
            )
        return int(pos)

    def get_multi_modal_attribute(self, attribute: str) -> 'DocumentArray':
        from docarray import DocumentArray

        position = self._get_mm_attr_postion(attribute)
        attribute_type = self._metadata['multi_modal_schema'][attribute][
            'attribute_type'
        ]

        if attribute_type in [AttributeType.DOCUMENT, AttributeType.NESTED]:
            return DocumentArray([self.chunks[position]])
        elif attribute_type in [
            AttributeType.ITERABLE_DOCUMENT,
            AttributeType.ITERABLE_NESTED,
        ]:
            return self.chunks[position].chunks
        else:
            raise ValueError(
                f'Invalid attribute {attribute}: must be a Document attribute or nested dataclass'
            )

    def set_multi_modal_attribute(
        self, attribute: str, value: typing.Union['Document', 'DocumentArray']
    ):
        position = self._get_mm_attr_postion(attribute)
        attribute_type = self._metadata['multi_modal_schema'][attribute][
            'attribute_type'
        ]

        if attribute_type in [AttributeType.DOCUMENT, AttributeType.NESTED]:
            self.chunks[position] = value
        elif attribute_type in [
            AttributeType.ITERABLE_DOCUMENT,
            AttributeType.ITERABLE_NESTED,
        ]:
            self.chunks[position].chunks = value
        else:
            raise ValueError(
                f'Invalid attribute {attribute}: must be a Document attribute or nested dataclass'
            )

    @classmethod
    def _from_obj(cls, obj, obj_type, field) -> typing.Tuple['Document', AttributeType]:
        attribute_type = AttributeType.DOCUMENT

        if is_multimodal(obj_type):
            doc = cls(obj)
            attribute_type = AttributeType.NESTED
        elif _is_field(field):
            doc = field.setter(obj)
        else:
            raise ValueError(f'Unsupported type annotation')
        return doc, attribute_type

    def _has_multimodal_attr(self, attr):
        try:
            data = super().__getattribute__('_data')
            has_data = bool(data)
        except AttributeError:
            return False

        has_metadata = has_data and getattr(self._data, '_metadata') is not None
        return (
            has_metadata
            and self.is_multimodal
            and attr in self._metadata['multi_modal_schema']
        )

    def __getattr__(self, attr):
        if self._has_multimodal_attr(attr):
            mm_attr_da = self.get_multi_modal_attribute(attr)
            return mm_attr_da if len(mm_attr_da) > 1 else mm_attr_da[0]
        else:
            raise AttributeError(f'{self.__class__.__name__} has no attribute {attr}')

    def __setattr__(self, attr, value):
        if self._has_multimodal_attr(attr):
            self.set_multi_modal_attribute(attr, value)
        else:
            object.__setattr__(self, attr, value)
