(mm-construct)=
# Construct

```{tip}
In DocArray, a Document object can contain sub-Document in `.chunks`. If you are still unaware of this design, make sure to read {ref}`this chapter<recursive-nested-document>` before continuing.
```


Just like the Python dataclasses module, DocArray provides a decorator {meth}`~docarray.dataclasses.types.dataclass` and a set of type annotations in {mod}`docarray.typing` such as `Image`, `Text`, `Audio`, that allow you to construct multimodal Document in the following way:

```python
from docarray import dataclass
from docarray.typing import Image, Text


@dataclass
class MyMultiModalDoc:
    avatar: Image
    description: Text


m = MyMultiModalDoc(avatar='test-1.jpeg', description='hello, world')
```

**Each field is a modality.** The above example contains two modalities: image and text.

```{Caution}

Be careful when assigning names to your modalities.
 
Do not use names that are properties of {class}`~docarray.document.Document`, such as
`text`, `tensor`, `embedding`, etc.
Instead, use more specific names that fit your domain, such as `avatar` and `description` in the example above.

If there is a conflict between the name of a modality and a property of {class}`~docarray.document.Document`,
no guarantees about the behavior while {ref}`accessing <mm-access-doc>` such a name can be made.

```

To convert it into a `Document` object, simply:

```python
from docarray import Document

d = Document(m)
d.summary()
```

This creates a Document object with two chunks:

````{dropdown} Nested structure (chunks)

```text
📄 Document: f3b193bbe8403c3ce1599b82f941f68a
└── 💠 Chunks
    ├── 📄 Document: 18c7ca1c829fe819250faa4914bc45c1
    │   ╭──────────────┬───────────────────────────────────────────────────────────────╮
    │   │ Attribute    │ Value                                                         │
    │   ├──────────────┼───────────────────────────────────────────────────────────────┤
    │   │ parent_id    │ f3b193bbe8403c3ce1599b82f941f68a                              │
    │   │ granularity  │ 1                                                             │
    │   │ tensor       │ <class 'numpy.ndarray'> in shape (504, 504, 3), dtype: uint8  │
    │   │ mime_type    │ image/jpeg                                                    │
    │   │ uri          │ test-1.jpeg                                                   │
    │   │ modality     │ image                                                         │
    │   ╰──────────────┴───────────────────────────────────────────────────────────────╯
    └── 📄 Document: 1ee7fadddc23fc72365b2069f82d4bb4
        ╭──────────────────────┬───────────────────────────────────────────────────────╮
        │ Attribute            │ Value                                                 │
        ├──────────────────────┼───────────────────────────────────────────────────────┤
        │ parent_id            │ f3b193bbe8403c3ce1599b82f941f68a                      │
        │ granularity          │ 1                                                     │
        │ text                 │ hello, world                                          │
        │ modality             │ text                                                  │
        ╰──────────────────────┴───────────────────────────────────────────────────────╯
```

````

To convert a Document object back to a `MyMultiModalDoc` object, do:

```python
m = MyMultiModalDoc(d)
```



## Dataclass decorator

First, you need to import `dataclass` decorator from DocArray package:

```python
from docarray import dataclass
```

The look and feel of this `dataclass` decorator is the same as the built-in `dataclass` decorator. In fact, any class wrapped by `docarray.dataclass` is also a valid Python `dataclass`:

```{code-block} python
---
emphasize-lines: 3, 6
---
from dataclasses import is_dataclass

from docarray import dataclass, Image


@dataclass
class MMDoc:
    banner: Image = 'test-1.jpeg'


print(is_dataclass(MMDoc))
print(is_dataclass(MMDoc()))
```


```text
True
True
```

That means, [arguments accepted by standard `dataclass`](https://docs.python.org/3/library/dataclasses.html#dataclasses.dataclass) are also accepted here. Methods that can be applied to Python `dataclass` can be also be applied to DocArray `dataclass`.

To tell if a class or object is DocArray's dataclass, you can use {meth}`~docarray.dataclasses.types.is_multimodal`:

```python
from docarray.typing import Image
import dataclasses

import docarray
from docarray.dataclasses import is_multimodal


@docarray.dataclass
class MMDoc1:
    banner: Image = 'test-1.jpeg'


@dataclasses.dataclass
class MMDoc2:
    banner: Image = 'test-1.jpeg'


print(is_multimodal(MMDoc1))
print(is_multimodal(MMDoc2))
```

```text
True
False
```


In the sequel, unless otherwise specified `dataclass` always refers to `docarray.dataclass`, not the Python built-in `dataclass`.


## Annotate class fields

DocArray provides {mod}`docarray.typing` that allows one to annotate a class field as `Image`, `Text`, `JSON`, `Audio`, `Video`, `Mesh`, `Tabular`, `Blob`; or as primitive Python types; or as other `docarray.dataclass`. 

```python
from docarray import dataclass
from docarray.typing import Image, Text, Audio


@dataclass
class MMDoc2:
    banner: Image = 'test-1.jpeg'
    summary: Text = 'This is an empty test image in 256 x 256 pixels'
    soundfx: Audio = 'white-noise.wav'
```

Convert `MMDoc2` object into a `Document` object is easy, simply via
```python
from docarray import Document

m = MMDoc2()
d = Document(m)
```

One can look at the structure of `d` via `d.summary()`:

````{dropdown} Nested structure (chunks)

```text
📄 Document: 90c744c5155c2356d27f8c91955f70f7
└── 💠 Chunks
    ├── 📄 Document: c9d71990088fb0d8db3c83a6bd35650d
    │   ╭──────────────┬───────────────────────────────────────────────────────────────╮
    │   │ Attribute    │ Value                                                         │
    │   ├──────────────┼───────────────────────────────────────────────────────────────┤
    │   │ parent_id    │ 90c744c5155c2356d27f8c91955f70f7                              │
    │   │ granularity  │ 1                                                             │
    │   │ tensor       │ <class 'numpy.ndarray'> in shape (504, 504, 3), dtype: uint8  │
    │   │ mime_type    │ image/jpeg                                                    │
    │   │ uri          │ test-1.jpeg                                                   │
    │   │ modality     │ image                                                         │
    │   ╰──────────────┴───────────────────────────────────────────────────────────────╯
    ├── 📄 Document: 22e27708288e70813673711c86f834ee
    │   ╭─────────────────┬────────────────────────────────────────────────────────────╮
    │   │ Attribute       │ Value                                                      │
    │   ├─────────────────┼────────────────────────────────────────────────────────────┤
    │   │ parent_id       │ 90c744c5155c2356d27f8c91955f70f7                           │
    │   │ granularity     │ 1                                                          │
    │   │ text            │ This is an empty test image in 256 x 256 pixels            │
    │   │ modality        │ text                                                       │
    │   ╰─────────────────┴────────────────────────────────────────────────────────────╯
    └── 📄 Document: 05ad36dfb0c520027b18c582d205c176
        ╭──────────────┬───────────────────────────────────────────────────────────────╮
        │ Attribute    │ Value                                                         │
        ├──────────────┼───────────────────────────────────────────────────────────────┤
        │ parent_id    │ 90c744c5155c2356d27f8c91955f70f7                              │
        │ granularity  │ 1                                                             │
        │ tensor       │ <class 'numpy.ndarray'> in shape (63248,), dtype: float32     │
        │ modality     │ audio                                                         │
        ╰──────────────┴───────────────────────────────────────────────────────────────╯
```

````

(mm-annotation)=
## Behavior of field annotation

This section explains the behavior of field annotations in details.

- A `dataclass` corresponds to a `Document` object, let's call it `root`.
- Unannotated fields are ignored.
    
    ````{tab} Field without type annotation
    
    ```python
    from docarray import dataclass, Document


    @dataclass
    class Doc:
        a = 1
        b = 'hello'


    Document(Doc()).summary()
    ```
    ````
    
    ````{tab} Document structure
    
    ```text
    📄 Document: 17c77b443471f9d752cbcc360174b65f
    ```
    
    ````
- A class field annotated as a Python primitive data type will be put into `root.tags`.
    
    ````{tab} Fields in primitive type
    
    ```python
    from docarray import dataclass, Document


    @dataclass
    class MMDoc:
        some_field: str = 'hello'
        other_field: int = 1


    Document(MMDoc()).summary()
    ``` 
    ````
    
    ````{tab} Document structure
    ```text
    📄 Document: 15725d705b6c8d7e99908d380d614fa5
    ╭────────────────┬─────────────────────────────────────────────────────────────╮
    │ Attribute      │ Value                                                       │
    ├────────────────┼─────────────────────────────────────────────────────────────┤
    │ tags           │ {'some_field': 'hello', 'other_field': 1}                   │
    ╰────────────────┴─────────────────────────────────────────────────────────────╯
    ```
    
    ````

- A class field annotated as a DocArray type will create a sub-Document nested under `root.chunks`.
    
    ````{tab} Field in DocArray type
    
    ```python
    from docarray import dataclass, Document
    from docarray.typing import Image


    @dataclass
    class MMDoc:
        some_field: str = 'hello'
        banner: Image = 'test-1.jpeg'


    Document(MMDoc()).summary()
    ```
    
    ````
    
    ````{tab} Document structure
    
    ```text
    📄 Document: 48a84621d51d94383b86db89e64022a3
    ╭────────────────────────┬─────────────────────────────────────────────────────╮
    │ Attribute              │ Value                                               │
    ├────────────────────────┼─────────────────────────────────────────────────────┤
    │ tags                   │ {'some_field': 'hello'}                             │
    ╰────────────────────────┴─────────────────────────────────────────────────────╯
    └── 💠 Chunks
        └── 📄 Document: 1cb5cc74f1f986876a0c4c87201b9a28
            ╭──────────────┬───────────────────────────────────────────────────────────────╮
            │ Attribute    │ Value                                                         │
            ├──────────────┼───────────────────────────────────────────────────────────────┤
            │ parent_id    │ 48a84621d51d94383b86db89e64022a3                              │
            │ granularity  │ 1                                                             │
            │ tensor       │ <class 'numpy.ndarray'> in shape (504, 504, 3), dtype: uint8  │
            │ mime_type    │ image/jpeg                                                    │
            │ uri          │ test-1.jpeg                                                   │
            │ modality     │ image                                                         │
            ╰──────────────┴───────────────────────────────────────────────────────────────╯
    ```
    
    ````
(type-list)=
- The annotation type determines how the sub-Document is constructed. For example, annotating a field as `Image` will instruct the construction to fill in `doc.tensor` by reading the image URI. Annotating a field as `JSON` will instruct the construction to fill in `doc.tags`. The complete behavior table can be found below:
    
| Type annotation | Accepted value types   | Behavior                                                                                                                                                                                       |
|-----------------|------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `Image`         | `str`, `numpy.ndarray` | Creates a sub-Document, fills in `doc.tensor` by reading the image and sets `.modality='image'`                                                                                                |
| `Text`          | `str`                  | Creates a sub-Document, fills in `doc.text` by the given value and sets `.modality='text'`                                                                                                     |
| `Audio`         | `str`, `numpy.ndarray` | Creates a sub-Document, fills in `doc.tensor` by reading the audio and sets `.modality='audio'`                                                                                                |
| `JSON`          | `Dict`                 | Creates a sub-Document, fills in `doc.tags` by the given value and sets `.modality='json'`                                                                                                     |
| `Video`         | `str`, `numpy.ndarray` | Creates a sub-Document, fills in `doc.tensor` by reading the video and sets `.modality='video'`                                                                                                |
| `Mesh`          | `str`, `numpy.ndarray` | Creates a sub-Document, fills in `doc.tensor` by sub-sampling the mesh as point-cloud and sets `.modality='mesh'`                                                                              |
| `Blob`          | `str`, `bytes`         | Creates a sub-Document, fills in `doc.blob` by the given value or reading from the path                                                                                                        |
| `Tabular`       | `str` (file name)      | Reads a CSV file, creates a sub-Document for each line and fills in `doc.tags` by considering the first row as the column names and mapping the following lines into the corresponding values. |

- A class field labeled with `List[Type]` will create sub-Documents under `root.chunks[0].chunks`. For example,
    
    ````{tab} Field in List[Image] 
  
    ```python
    from typing import List

    from docarray import dataclass, Document
    from docarray.typing import Image


    @dataclass
    class MMDoc2:
        banners: List[Image]


    Document(MMDoc2(['test-1.jpeg', 'test-2.jpeg'])).summary()
    ```
    ````
    
    ````{tab} Document structure
  
    ```text
    📄 Document: 52d9dcca4bc30cd0ef3b82917459cd32
    └── 💠 Chunks
        └── 📄 Document: 04edacf582c5aa7b0bcfcf3d3e0a57bf
            ╭──────────────────────┬───────────────────────────────────────────────────────╮
            │ Attribute            │ Value                                                 │
            ├──────────────────────┼───────────────────────────────────────────────────────┤
            │ parent_id            │ 52d9dcca4bc30cd0ef3b82917459cd32                      │
            │ granularity          │ 1                                                     │
            ╰──────────────────────┴───────────────────────────────────────────────────────╯
            └── 💠 Chunks
                ├── 📄 Document: f5e9f105162e26d1d42ef7e2d363095a
                │   ╭──────────────┬───────────────────────────────────────────────────────────────╮
                │   │ Attribute    │ Value                                                         │
                │   ├──────────────┼───────────────────────────────────────────────────────────────┤
                │   │ parent_id    │ 04edacf582c5aa7b0bcfcf3d3e0a57bf                              │
                │   │ granularity  │ 1                                                             │
                │   │ tensor       │ <class 'numpy.ndarray'> in shape (504, 504, 3), dtype: uint8  │
                │   │ mime_type    │ image/jpeg                                                    │
                │   │ uri          │ test-1.jpeg                                                   │
                │   │ modality     │ image                                                         │
                │   ╰──────────────┴───────────────────────────────────────────────────────────────╯
                └── 📄 Document: d7d0b506f690890113e6a601ef80f8c6
                    ╭──────────────┬───────────────────────────────────────────────────────────────╮
                    │ Attribute    │ Value                                                         │
                    ├──────────────┼───────────────────────────────────────────────────────────────┤
                    │ parent_id    │ 04edacf582c5aa7b0bcfcf3d3e0a57bf                              │
                    │ granularity  │ 1                                                             │
                    │ tensor       │ <class 'numpy.ndarray'> in shape (504, 504, 3), dtype: uint8  │
                    │ mime_type    │ image/jpeg                                                    │
                    │ uri          │ test-2.jpeg                                                   │
                    │ modality     │ image                                                         │
                    ╰──────────────┴───────────────────────────────────────────────────────────────╯
    ```
    ````
- A field annotated with another `dataclass` will create the full nested structure under the corresponding chunk.
    
    ````{tab} Field in another dataclass
    
    ```python
    from docarray import dataclass, Document
    from docarray.typing import Image, Text


    @dataclass
    class BannerDoc:
        description: Text = 'this is a test empty image'
        banner: Image = 'test-1.jpeg'


    @dataclass
    class ColumnArticle:
        feature_image: BannerDoc
        description: Text = 'this is a column article'
        website: str = 'https://jina.ai'


    Document(ColumnArticle(feature_image=BannerDoc())).summary()
    ```
    
    ````
    ````{tab} Document structure
    ```text
    📄 Document: 75a3df4c26498d338589d2b2c20e156e
    ╭────────────────────┬─────────────────────────────────────────────────────────╮
    │ Attribute          │ Value                                                   │
    ├────────────────────┼─────────────────────────────────────────────────────────┤
    │ tags               │ {'website': 'https://jina.ai'}                          │
    ╰────────────────────┴─────────────────────────────────────────────────────────╯
    └── 💠 Chunks
        ├── 📄 Document: cb1df29a384a6d39aa81e5af93316c4d
        │   ╭──────────────────────┬───────────────────────────────────────────────────────╮
        │   │ Attribute            │ Value                                                 │
        │   ├──────────────────────┼───────────────────────────────────────────────────────┤
        │   │ parent_id            │ 75a3df4c26498d338589d2b2c20e156e                      │
        │   │ granularity          │ 1                                                     │
        │   ╰──────────────────────┴───────────────────────────────────────────────────────╯
        │   └── 💠 Chunks
        │       ├── 📄 Document: 65cce8eb564f9ce136ff693b8ecb8f53
        │       │   ╭──────────────────────┬───────────────────────────────────────────────────────╮
        │       │   │ Attribute            │ Value                                                 │
        │       │   ├──────────────────────┼───────────────────────────────────────────────────────┤
        │       │   │ parent_id            │ cb1df29a384a6d39aa81e5af93316c4d                      │
        │       │   │ granularity          │ 1                                                     │
        │       │   │ text                 │ this is a test empty image                            │
        │       │   │ modality             │ text                                                  │
        │       │   ╰──────────────────────┴───────────────────────────────────────────────────────╯
        │       └── 📄 Document: 4dc4497d608b4f96094711e90cfb8078
        │           ╭──────────────┬───────────────────────────────────────────────────────────────╮
        │           │ Attribute    │ Value                                                         │
        │           ├──────────────┼───────────────────────────────────────────────────────────────┤
        │           │ parent_id    │ cb1df29a384a6d39aa81e5af93316c4d                              │
        │           │ granularity  │ 1                                                             │
        │           │ tensor       │ <class 'numpy.ndarray'> in shape (504, 504, 3), dtype: uint8  │
        │           │ mime_type    │ image/jpeg                                                    │
        │           │ uri          │ test-1.jpeg                                                   │
        │           │ modality     │ image                                                         │
        │           ╰──────────────┴───────────────────────────────────────────────────────────────╯
        └── 📄 Document: f7b3aaefeab73af18f8372a594405b46
            ╭──────────────────────┬───────────────────────────────────────────────────────╮
            │ Attribute            │ Value                                                 │
            ├──────────────────────┼───────────────────────────────────────────────────────┤
            │ parent_id            │ 75a3df4c26498d338589d2b2c20e156e                      │
            │ granularity          │ 1                                                     │
            │ text                 │ this is a column article                              │
            │ modality             │ text                                                  │
            ╰──────────────────────┴───────────────────────────────────────────────────────╯
    
    ```
      
    ````
- A dataclass that has only one field annotated with `docarray.typing` will still create a nested structure under `root.chunks`. In this case, `len(root.chunks)=1` and your multimodal Document has basically a single modality, which may encourage you to think if this is really necessary to use a `dataclass`. After all, each Document represents single modality, and you can just use `Document`.  

## Construct from/to Document

It is easy to convert a `dataclass` object from/to a `Document` object:

```python
from docarray import dataclass, Document
from docarray.typing import Image


@dataclass
class MMDoc:
    banner: Image


m = MMDoc(banner='test-0.jpeg')
d = Document(m)  # to Document object
m_r = MMDoc(d)  # from Document object

assert m == m_r
```

## Use `field()` for advanced configs

For common and simple use cases, no other functionality is required. There are, however, some dataclass features that require additional per-field information. To satisfy this need for additional information, you can replace the default field value with a call to the provided {meth}`~docarray.dataclasses.types.field` function.

For example, mutable object is not allowed as the default value of any dataclass field. One can solve it via:

```python
from typing import List

from docarray import dataclass, field
from docarray.typing import Image


@dataclass
class MMDoc:
    banner: List[Image] = field(default_factory=lambda: ['test-1.jpeg', 'test-2.jpeg'])
```

Other parameters from the standard the Python field such as `init`, `compare`, `hash`, `repr` are also supported. More details can be [found here](https://docs.python.org/3/library/dataclasses.html#dataclasses.field).


## What's next?

In this chapter, we have learned to use `@dataclass` decorator and type annotation to build multimodal documents. The look and feel is exactly the same as Python builtin dataclass.   

Leveraging {ref}`the nested Document structure<recursive-nested-document>`, DocArray's dataclass offers great expressiveness for data scientists and machine learning engineers who work with multimodal data, allowing them to represent image, text, video, mesh, tabular data in a very intuitive way. Converting a multimodal dataclass object from/to a Document is very straightforward. 

In the next chapter, we shall see how to select modality (aka sub-document) via the selector syntax.