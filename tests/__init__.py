import numpy as np
from docarray import DocumentArray, Document


def random_docs(
    num_docs,
    chunks_per_doc=5,
    embed_dim=10,
    jitter=1,
    start_id=0,
    embedding=True,
    sparse_embedding=False,
    text='hello world',
) -> DocumentArray:
    da = DocumentArray()
    next_chunk_doc_id = start_id + num_docs
    for j in range(num_docs):
        doc_id = str(start_id + j)

        d = Document(id=doc_id)
        d.text = text
        d.tags['id'] = f'myself id is: {doc_id}'
        if embedding:
            if sparse_embedding:
                from scipy.sparse import coo_matrix

                d.embedding = coo_matrix(
                    (np.array([1, 1, 1]), (np.array([0, 1, 2]), np.array([1, 2, 1])))
                )
            else:
                d.embedding = np.random.random(
                    [embed_dim + np.random.randint(0, jitter)]
                )

        for _ in range(chunks_per_doc):
            chunk_doc_id = str(next_chunk_doc_id)

            c = Document(id=chunk_doc_id)
            c.text = 'i\'m chunk %s from doc %s' % (chunk_doc_id, doc_id)
            if embedding:
                c.embedding = np.random.random(
                    [embed_dim + np.random.randint(0, jitter)]
                )
            c.tags['parent_id'] = f'my parent is: {id}'
            c.tags['id'] = f'myself id is: {doc_id}'
            d.chunks.append(c)
            next_chunk_doc_id += 1

        da.append(d)
    return da
