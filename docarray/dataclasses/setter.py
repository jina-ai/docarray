from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from docarray import Document


def image_setter(value) -> 'Document':
    from docarray import Document

    doc = Document(modality='image')

    if isinstance(value, str):
        doc.uri = value
        doc._metadata['image_type'] = 'uri'
        doc.load_uri_to_image_tensor()
    elif isinstance(value, np.ndarray):
        doc.tensor = value
        doc._metadata['image_type'] = 'ndarray'
    else:
        from PIL.Image import Image

        if isinstance(value, Image):
            doc.tensor = np.array(value)
            doc._metadata['image_type'] = 'PIL'
    return doc


def image_setter_(width, height, channel_axis, value) -> 'Document':
    from docarray import Document

    doc = Document(modality='image')

    if isinstance(value, str):
        doc.uri = value
        doc._metadata['image_type'] = 'uri'
        doc.load_uri_to_image_tensor(width, height, channel_axis)
    elif isinstance(value, np.ndarray):
        doc.tensor = value
        doc._metadata['image_type'] = 'ndarray'
    else:
        from PIL.Image import Image

        if isinstance(value, Image):
            doc.tensor = np.array(value)
            doc._metadata['image_type'] = 'PIL'
    return doc


def text_setter(value) -> 'Document':
    from docarray import Document

    return Document(text=value, modality='text')


def audio_setter(value) -> 'Document':
    from docarray import Document

    if isinstance(value, np.ndarray):
        return Document(tensor=value, _metadata={'audio_type': 'ndarray'})
    else:
        return Document(
            uri=value, modality='audio', _metadata={'audio_type': 'uri'}
        ).load_uri_to_audio_tensor()


def video_setter(value) -> 'Document':
    from docarray import Document

    if isinstance(value, np.ndarray):
        return Document(tensor=value, _metadata={'video_type': 'ndarray'})
    else:
        return Document(
            uri=value, modality='video', _metadata={'video_type': 'uri'}
        ).load_uri_to_video_tensor()


def mesh_setter(value) -> 'Document':
    from docarray import Document

    if isinstance(value, np.ndarray):
        return Document(tensor=value, _metadata={'mesh_type': 'ndarray'})
    else:
        return Document(
            uri=value, modality='mesh', _metadata={'mesh_type': 'uri'}
        ).load_uri_to_point_cloud_tensor(1000)


def blob_setter(value) -> 'Document':
    from docarray import Document

    if isinstance(value, bytes):
        return Document(blob=value, _metadata={'blob_type': 'bytes'})
    else:
        return Document(uri=value, _metadata={'blob_type': 'uri'}).load_uri_to_blob()


def json_setter(value) -> 'Document':
    from docarray import Document

    return Document(modality='json', tags=value)


def tabular_setter(value) -> 'Document':
    from docarray import Document, DocumentArray

    return Document(uri=value, chunks=DocumentArray.from_csv(value), modality='tabular')
