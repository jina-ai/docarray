(text-type)=
# {octicon}`typography` Text

Representing text in DocArray is easy. Simply do:
```python
from docarray import Document

Document(text='hello, world.')
```

If your text data is big and can not be written inline, or it comes from a URI, then you can also define `uri` first and load the text into Document later.

```python
from docarray import Document

d = Document(uri='https://www.w3.org/History/19921103-hypertext/hypertext/README.html')
d.load_uri_to_text()

d.summary()
```

```text
<Document ('id', 'mime_type', 'text', 'uri') at 3c128f326fbf11ec90821e008a366d49>
```

And of course, you can have characters from different languages.

```python
from docarray import Document

d = Document(text='👋	नमस्ते दुनिया!	你好世界！こんにちは世界！	Привет мир!')
```


## Segment long documents

Often times when you index/search textual document, you don't want to consider thousands of words as one document, some finer granularity would be nice. You can do these by leveraging `chunks` of Document. For example, let's segment this simple document by `!` mark:

```python
from docarray import Document

d = Document(text='👋	नमस्ते दुनिया!	你好世界!こんにちは世界!	Привет мир!')

d.chunks.extend([Document(text=c) for c in d.text.split('!')])

d.summary()
```

```text
 <Document ('id', 'mime_type', 'text', 'chunks') at 5a12d7a86fbf11ec99a21e008a366d49>
    └─ chunks
          ├─ <Document ('id', 'mime_type', 'text') at 5a12e2346fbf11ec99a21e008a366d49>
          ├─ <Document ('id', 'mime_type', 'text') at 5a12e2f26fbf11ec99a21e008a366d49>
          ├─ <Document ('id', 'mime_type', 'text') at 5a12e3886fbf11ec99a21e008a366d49>
          ├─ <Document ('id', 'mime_type', 'text') at 5a12e41e6fbf11ec99a21e008a366d49>
          └─ <Document ('id',) at 5a12e4966fbf11ec99a21e008a366d49>
```

Which creates five sub-documents under the original documents and stores them under `.chunks`.

## Convert text into `ndarray`

Sometimes you may need to encode the text into a `numpy.ndarray` before further computation. We provide some helper functions in Document and DocumentArray that allow you to convert easily.

For example, we have a DocumentArray with three Documents:
```python
from docarray import DocumentArray, Document

da = DocumentArray([Document(text='hello world'), 
                    Document(text='goodbye world'),
                    Document(text='hello goodbye')])
```

To get the vocabulary, you can use:

```python
vocab = da.get_vocabulary()
```

```text
{'hello': 2, 'world': 3, 'goodbye': 4}
```

The vocabulary is 2-indexed as `0` is reserved for padding symbol and `1` is reserved for unknown symbol.

One can further use this vocabulary to convert `.text` field into `.tensor` via:

```python
for d in da:
    d.convert_text_to_tensor(vocab)
    print(d.tensor)
```

```text
[2 3]
[4 3]
[2 4]
```

When you have text in different length and you want the output `.tensor` to have the same length, you can define `max_length` during converting:

```python
from docarray import Document, DocumentArray

da = DocumentArray([Document(text='a short phrase'), 
                    Document(text='word'), 
                    Document(text='this is a much longer sentence')])
vocab = da.get_vocabulary()

for d in da:
    d.convert_text_to_tensor(vocab, max_length=10)
    print(d.tensor)
```

```text
[0 0 0 0 0 0 0 2 3 4]
[0 0 0 0 0 0 0 0 0 5]
[ 0  0  0  0  6  7  2  8  9 10]
```

You can get also use `.tensors` of DocumentArray to get all tensors in one `ndarray`.

```python
print(da.tensors)
````

```text
[[ 0  0  0  0  0  0  0  2  3  4]
 [ 0  0  0  0  0  0  0  0  0  5]
 [ 0  0  0  0  6  7  2  8  9 10]]
```

## Convert `ndarray` back to text

As a bonus, you can also easily convert an integer `ndarray` back to text based on some given vocabulary. This procedure is often termed as "decoding". 

```python
from docarray import Document, DocumentArray

da = DocumentArray([Document(text='a short phrase'), 
                    Document(text='word'), 
                    Document(text='this is a much longer sentence')])
vocab = da.get_vocabulary()

# encoding
for d in da:
    d.convert_text_to_tensor(vocab, max_length=10)

# decoding
for d in da:
    d.convert_tensor_to_text(vocab)
    print(d.text)
```

```text
a short phrase
word
this is a much longer sentence
```


## Simple text matching via feature hashing

Let's search for `"she entered the room"` in *Pride and Prejudice*:

```python
from docarray import Document, DocumentArray

d = Document(uri='https://www.gutenberg.org/files/1342/1342-0.txt').load_uri_to_text()
da = DocumentArray(Document(text=s.strip()) for s in d.text.split('\n') if s.strip())
da.apply(lambda d: d.embed_feature_hashing())

q = (
    Document(text='she entered the room')
    .embed_feature_hashing()
    .match(da, limit=5, exclude_self=True, metric='jaccard', use_scipy=True)
)

print(q.matches[:, ('text', 'scores__jaccard')])
```

```text
[['staircase, than she entered the breakfast-room, and congratulated', 
'of the room.', 
'She entered the room with an air more than usually ungracious,', 
'entered the breakfast-room, where Mrs. Bennet was alone, than she', 
'those in the room.'], 
[{'value': 0.6, 'ref_id': 'f47f7448709811ec960a1e008a366d49'}, 
{'value': 0.6666666666666666, 'ref_id': 'f47f7448709811ec960a1e008a366d49'}, 
{'value': 0.6666666666666666, 'ref_id': 'f47f7448709811ec960a1e008a366d49'}, 
{'value': 0.6666666666666666, 'ref_id': 'f47f7448709811ec960a1e008a366d49'}, 
{'value': 0.7142857142857143, 'ref_id': 'f47f7448709811ec960a1e008a366d49'}]]
```


## Searching at chunk level with secondary index

One can create applications that search at chunk level using a secondary index. 
Imagine you want an application that searches at a sentences granularity and returns the document title of the document
containing the sentence closest to the query. For example, you can have a database of lyrics of songs and you want to
search the song title of a song from which you might remember a small part of it (likely the chorus).

```python
song1_title = 'Take On Me' 

song1 = """
#A-ha
Talking away
I don't know what I'm to say
I'll say it anyway
Today is another day to find you
Shying away
I'll be coming for your love. OK?

Take on me (take on me)
Take me on (take on me)
I'll be gone
In a day or two

So needless to say
Of odds and ends
But I'll be stumbling away
Slowly learning that life is OK.
Say after me,
"It's no better to be safe than sorry."

Take on me (take on me)
Take me on (take on me)
I'll be gone
In a day or two

Oh, things that you say. Yeah.
Is it life or just to play my worries away?
You're all the things I've got to remember
You're shying away
I'll be coming for you anyway

Take on me (take on me)
Take me on (take on me)
I'll be gone
In a day
"""

song2_title = 'The trooper' 

song2 = """
You'll take my life, but I'll take yours too
You'll fire your musket, but I'll run you through
So when you're waiting for the next attack
You'd better stand, there's no turning back
The bugle sounds, the charge begins
But on this battlefield, no one wins
The smell of acrid smoke and horses' breath
As I plunge on into certain death
The horse, he sweats with fear, we break to run
The mighty roar of the Russian guns
And as we race towards the human wall
The screams of pain as my comrades fall
We hurdle bodies that lay on the ground
And the Russians fire another round
We get so near, yet so far away
We won't live to fight another day
We get so close, near enough to fight
When a Russian gets me in his sights
He pulls the trigger and I feel the blow
A burst of rounds take my horse below
And as I lay there gazing at the sky
My body's numb and my throat is dry
And as I lay forgotten and alone
Without a tear, I draw my parting groan
"""
```

We can now create one document for each of the songs, containing as chunks the song sentences.

```python
from docarray import Document, DocumentArray
doc1 = Document(chunks = [ Document(text=line) for line in song1.split('\n')], song_title = song1_title)
doc2 = Document(chunks = [ Document(text=line) for line in song2.split('\n')], song_title = song2_title)
da = DocumentArray()
da.extend([doc1, doc2])
```

Now we can build a feature vector for each line of each song. Here we use a very simple Bag of Words descriptor as 
feature vector.

```python
import re

def build_tokenizer(token_pattern=r"(?u)\b\w\w+\b"):
    token_pattern = re.compile(token_pattern)
    return token_pattern.findall

def bow_feature_vector(d, vocab, tokenizer):
    embedding = np.zeros(len(vocab)+2)
    tokens = tokenizer(d.text)
    for token in tokens:
        if token in vocab:
            embedding[vocab.get(token)] +=1
        
    return embedding


tokenizer = build_tokenizer()
vocab = da['@c'].get_vocabulary()
for d in da['@c']:
    d.embedding=bow_feature_vector(d, vocab, tokenizer)
```

Once we have the data prepared, we can store it into a DocumentArray that supports a secondary index.

```buildoutcfg
n_features = len(vocab)+2
n_dim = 3
da_backend=DocumentArray(
    storage='annlite',
    config={'data_path':'./annlite_data',
            'n_dim': n_dim, 
            'metric': 'Cosine'},
    secondary_indices_configs={'@c': {'n_dim': n_features}},
)

with da_backend:
    da_backend.extend(da)
```

Given a query such as `into death` we want to search which song contained a similar sentence.

```python
def find_song_name_from_song_snippet(query: Document,
                                     da_backend) -> str:
    similar_items = da_backend.find(query=query,
                                    secondary_index = '@c',
                                    limit=10)[0]
    most_similar_docs = similar_items[0]
    return da_backend[most_similar_docs.parent_id].tags

query = Document(text='into death')
query.embedding = bow_feature_vector(query, vocab, tokenizer)

similar_items = find_song_name_from_song_snippet(query, da_backend)
print(similar_items)
```
Will print 
```text
{'song_title': 'The trooper'}
```
