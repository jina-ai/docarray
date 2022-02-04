import os

import numpy as np
import tensorflow as tf
import pytest
import torch
import paddle
import onnx
import onnxruntime
from transformers import ViTModel, TFViTModel, ViTConfig

from docarray import DocumentArray
from docarray.array.memory import DocumentArrayInMemory
from docarray.array.sqlite import DocumentArraySqlite
from docarray.array.weaviate import DocumentArrayWeaviate

random_embed_models = {
    'keras': lambda: tf.keras.Sequential(
        [tf.keras.layers.Dropout(0.5), tf.keras.layers.BatchNormalization()]
    ),
    'pytorch': lambda: torch.nn.Sequential(
        torch.nn.Dropout(0.5), torch.nn.BatchNorm1d(128)
    ),
    'paddle': lambda: paddle.nn.Sequential(
        paddle.nn.Dropout(0.5), paddle.nn.BatchNorm1D(128)
    ),
    'transformers_torch': lambda: ViTModel(ViTConfig()),
    'transformers_tf': lambda: TFViTModel(ViTConfig()),
}
cur_dir = os.path.dirname(os.path.abspath(__file__))
torch.onnx.export(
    random_embed_models['pytorch'](),
    torch.rand(1, 128),
    os.path.join(cur_dir, 'test-net.onnx'),
    do_constant_folding=True,  # whether to execute constant folding for optimization
    input_names=['input'],  # the model's input names
    output_names=['output'],  # the model's output names
    dynamic_axes={
        'input': {0: 'batch_size'},  # variable length axes
        'output': {0: 'batch_size'},
    },
)

random_embed_models['onnx'] = lambda: onnxruntime.InferenceSession(
    os.path.join(cur_dir, 'test-net.onnx')
)


@pytest.mark.parametrize('framework', ['onnx', 'keras', 'pytorch', 'paddle'])
@pytest.mark.parametrize(
    'da', [DocumentArray, DocumentArraySqlite, DocumentArrayWeaviate]
)
@pytest.mark.parametrize('N', [2, 10])
@pytest.mark.parametrize('batch_size', [1, 256])
@pytest.mark.parametrize('to_numpy', [True, False])
def test_embedding_on_random_network(
    framework, da, N, batch_size, to_numpy, start_weaviate
):
    docs = da.empty(N)
    docs.tensors = np.random.random([N, 128]).astype(np.float32)
    embed_model = random_embed_models[framework]()
    docs.embed(embed_model, batch_size=batch_size, to_numpy=to_numpy)

    r = docs.embeddings
    if hasattr(r, 'numpy'):
        r = r.numpy()

    embed1 = r.copy()

    # reset
    docs.embeddings = np.random.random([N, 128]).astype(np.float32)

    # docs[a: b].embed is only supported for DocumentArrayInMemory
    if isinstance(da, DocumentArrayInMemory):
        # try it again, it should yield the same result
        docs.embed(embed_model, batch_size=batch_size, to_numpy=to_numpy)
        np.testing.assert_array_almost_equal(docs.embeddings, embed1)

        # reset
        docs.embeddings = np.random.random([N, 128]).astype(np.float32)

        # now do this one by one
        docs[: int(N / 2)].embed(embed_model, batch_size=batch_size, to_numpy=to_numpy)
        docs[-int(N / 2) :].embed(embed_model, batch_size=batch_size, to_numpy=to_numpy)
        np.testing.assert_array_almost_equal(docs.embeddings, embed1)


@pytest.mark.parametrize('framework', ['transformers_torch', 'transformers_tf'])
@pytest.mark.parametrize(
    'da', [DocumentArray, DocumentArraySqlite, DocumentArrayWeaviate]
)
@pytest.mark.parametrize('N', [2, 10])
@pytest.mark.parametrize('batch_size', [1, 256])
@pytest.mark.parametrize('to_numpy', [True, False])
def test_embedding_on_transformers(
    framework, da, N, batch_size, to_numpy, start_weaviate
):
    docs = da.empty(N)
    docs.tensors = np.random.random([N, 3, 224, 224]).astype(np.float32)
    embed_model = random_embed_models[framework]()
    docs.embed(embed_model, batch_size=batch_size, to_numpy=to_numpy)

    r = docs.embeddings
    if hasattr(r, 'numpy'):
        r = r.numpy()

    embed1 = r.copy()

    # reset
    docs.embeddings = np.random.random([N, 128]).astype(np.float32)

    # docs[a: b].embed is only supported for DocumentArrayInMemory
    if isinstance(da, DocumentArrayInMemory):
        # try it again, it should yield the same result
        docs.embed(embed_model, batch_size=batch_size, to_numpy=to_numpy)
        np.testing.assert_array_almost_equal(docs.embeddings, embed1)

        # reset
        docs.embeddings = np.random.random([N, 128]).astype(np.float32)

        # now do this one by one
        docs[: int(N / 2)].embed(embed_model, batch_size=batch_size, to_numpy=to_numpy)
        docs[-int(N / 2) :].embed(embed_model, batch_size=batch_size, to_numpy=to_numpy)
        np.testing.assert_array_almost_equal(docs.embeddings, embed1)
