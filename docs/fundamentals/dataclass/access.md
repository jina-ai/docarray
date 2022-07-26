# Access Modality

After {ref}`constructing a multi-modal Document or DocumentArray <mm-construct>`, you can directly access your custom-defined
modalities by their names.

```{admonition} Return types
:class: seealso

Accessing a modality always returns a Document or a DocumentArray, instead of directly returning the data stored in them.

If you are interested in the principles behind this design decision, you can learn more [here](TODO LINK TO BLOG POST).
```

(mm-access-doc)=
## Document level access

Even after conversion to {class}`~docarray.document.Document`, custom-defines modalities can be accessed by their names, returning a
{class}`~docarray.document.Document` or, for list-types, a {class}`~docarray.array.document.DocumentArray`.

```python
from docarray import Document, dataclass
from typing import List
from docarray.typing import Image, Text


@dataclass
class MMDoc:
    banner: Image
    paragraphs: List[Text]


doc = Document(
    MMDoc(
        banner='test.jpg',
        paragraphs=['This is a paragraph', 'this is another paragraph'],
    )
)

print(doc.banner)  # returns a Document with the test.jpg image tensor
print(doc.banner.tensor)  # returns the image tensor
print(doc.paragraphs)  # returns a DocumentArray with one Document per paragraph
print(doc.paragraphs.texts)  # returns the paragraph texts
```


```text
<Document ('id', 'parent_id', 'granularity', 'tensor', 'mime_type', 'uri', '_metadata', 'modality') at eaccc9c573c07f13b7ee8aa04a83c9eb>
[[[255 255 255]
  [255 255 255]
  [255 255 255]]]
<DocumentArray (length=2) at 140540453339296>
['This is a paragraph', 'this is another paragraph']
```

The returned Documents (or DocumentArrays) can be directly used to store additional information about the modality:

```python
import torch, torchvision

model = torchvision.models.resnet50(pretrained=True)
banner_tensor = torch.tensor(doc.banner.tensor).transpose(0, 2).unsqueeze(0)
doc.banner.embedding = model(banner_tensor)
```

### Select nested fields

Nested field, coming from {ref}`nested dataclasses <mm-annotation>`, can be accessed by selecting the outer field,
and then selecting the inner field:

```python
from docarray import dataclass, Document
from docarray.typing import Image, Text


@dataclass
class InnerDoc:
    description: Text
    banner: Image = 'test-1.jpeg'


@dataclass
class OuterDoc:
    feature_image: InnerDoc
    website: str = 'https://jina.ai'


doc = Document(OuterDoc(feature_image=InnerDoc(description='this is a description')))
print(
    doc.feature_image.description
)  # returns a Document with 'this is a description' as text
print(doc.feature_image.description.text)  # returns 'this is a description'
```

```text
<Document ('id', 'parent_id', 'granularity', 'text', 'modality') at 94de1bef2fc8010ff4fe86791a671c44>
this is a description
```

## DocumentArray level access

Custom modalities can be accessed through the familiar {ref}`selector syntax <access-elements>`.

Like all selectors, a selector for a multi-modal attribute begins with `@`.
The fact that a custom modality is accessed is denoted through the addition of a `.`, coming before a list of modality names:

```text
@.[field1, field2, ...]
^^ ~~~~~~  ~~~~~~
||   |       |
||   |-------|
||       |
||       | --- indicate the field of dataclass (modality name)
||
|| ------ indicate the start of modality selector
|
| ---- indicate the start of selector
```

Selecting a modality form a DocumentArray always results in another DocumentArray:

```python
from docarray import Document, dataclass, DocumentArray
from docarray.typing import Image, Text


@dataclass
class MMDoc:
    banner: Image
    description: Text


da = DocumentArray(
    [
        Document(
            MMDoc(banner='test-1.jpeg', description='this is a test white-noise image')
        ),
        Document(
            MMDoc(banner='test-2.jpeg', description='another test image but in black')
        ),
    ]
)

da.summary()
```

`````{tab} Select Documents corresponding to .banner 

```python
da['@.[banner]']
```
````{dropdown} Output
```text
╭───────────────────────────── Documents Summary ──────────────────────────────╮
│                                                                              │
│   Length                 2                                                   │
│   Homogenous Documents   True                                                │
│   Common Attributes      ('id', 'parent_id', 'granularity', 'tensor',        │
│                          'mime_type', 'uri', 'modality')                     │
│   Multimodal dataclass   False                                               │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─────────────────────── Attributes Summary ────────────────────────╮
│                                                                   │
│   Attribute     Data type      #Unique values   Has empty value   │
│  ───────────────────────────────────────────────────────────────  │
│   granularity   ('int',)       1                False             │
│   id            ('str',)       2                False             │
│   mime_type     ('str',)       1                False             │
│   modality      ('str',)       1                False             │
│   parent_id     ('str',)       2                False             │
│   tensor        ('ndarray',)   2                False             │
│   uri           ('str',)       2                False             │
│                                                                   │
╰───────────────────────────────────────────────────────────────────╯
```

````


`````

`````{tab} Select Documents corresponding to .description 

```python
da['@.[description]']
```

````{dropdown} Output

```text
╭───────────────────────────── Documents Summary ──────────────────────────────╮
│                                                                              │
│   Length                 2                                                   │
│   Homogenous Documents   True                                                │
│   Common Attributes      ('id', 'parent_id', 'granularity', 'text',          │
│                          'modality')                                         │
│   Multimodal dataclass   False                                               │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯
╭────────────────────── Attributes Summary ──────────────────────╮
│                                                                │
│   Attribute     Data type   #Unique values   Has empty value   │
│  ────────────────────────────────────────────────────────────  │
│   granularity   ('int',)    1                False             │
│   id            ('str',)    2                False             │
│   modality      ('str',)    1                False             │
│   parent_id     ('str',)    2                False             │
│   text          ('str',)    2                False             │
│                                                                │
╰────────────────────────────────────────────────────────────────╯
```

````

`````

### Select multiple fields

You can select multiple fields by including them in the square brackets, separated by a comma `,`:

```python
da['@.[description, banner]']
```

````{dropdown} Output


```text
╭───────────────────────────── Documents Summary ──────────────────────────────╮
│                                                                              │
│   Length                        4                                            │
│   Homogenous Documents          False                                        │
│   2 Documents have attributes   ('id', 'parent_id', 'granularity', 'text',   │
│                                 'modality')                                  │
│   2 Documents have attributes   ('id', 'parent_id', 'granularity',           │
│                                 'tensor', 'mime_type', 'uri', 'modality')    │
│   Multimodal dataclass          False                                        │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯
╭───────────────────────────── Attributes Summary ─────────────────────────────╮
│                                                                              │
│   Attribute     Data type                 #Unique values   Has empty value   │
│  ──────────────────────────────────────────────────────────────────────────  │
│   granularity   ('int',)                  1                False             │
│   id            ('str',)                  4                False             │
│   mime_type     ('str',)                  2                False             │
│   modality      ('str',)                  2                False             │
│   parent_id     ('str',)                  2                False             │
│   tensor        ('ndarray', 'NoneType')   4                True              │
│   text          ('str',)                  3                False             │
│   uri           ('str',)                  3                False             │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯
```
````

### Slice dataclass objects

Remember each dataclass object corresponds to one Document object, you can first slice the DocumentArray before selecting the field. Specifically, you can do:

```text
@r[slice].[field1, field2, ...]
```

where `slice` can be any slice syntax accepted in {ref}`access-elements`.

For example, to select the sub-Document `.banner` for only the first Document:
 

```python
da['@r[:1].[banner]']
```


````{dropdown} Output

```text
╭───────────────────────────── Documents Summary ──────────────────────────────╮
│                                                                              │
│   Length                 1                                                   │
│   Homogenous Documents   True                                                │
│   Common Attributes      ('id', 'parent_id', 'granularity', 'tensor',        │
│                          'mime_type', 'uri', 'modality')                     │
│   Multimodal dataclass   False                                               │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─────────────────────── Attributes Summary ────────────────────────╮
│                                                                   │
│   Attribute     Data type      #Unique values   Has empty value   │
│  ───────────────────────────────────────────────────────────────  │
│   granularity   ('int',)       1                False             │
│   id            ('str',)       1                False             │
│   mime_type     ('str',)       1                False             │
│   modality      ('str',)       1                False             │
│   parent_id     ('str',)       1                False             │
│   tensor        ('ndarray',)   1                False             │
│   uri           ('str',)       1                False             │
│                                                                   │
╰───────────────────────────────────────────────────────────────────╯

```

````

### Slice `List[Type]` fields

If a field is annotated as a List of DocArray types, it will create a DocumentArray, one can add slicing after the field selector to further restrict the size of the sub-Documents.

```{code-block} python
---
emphasize-lines: 30
---
from typing import List

from docarray import Document, dataclass, DocumentArray
from docarray.typing import Image, Text


@dataclass
class MMDoc:
    banner: List[Image]
    description: Text


da = DocumentArray(
    [
        Document(
            MMDoc(
                banner=['test-1.jpeg', 'test-2.jpeg'],
                description='this is a test white image',
            )
        ),
        Document(
            MMDoc(
                banner=['test-1.jpeg', 'test-2.jpeg'],
                description='another test image but in black',
            )
        ),
    ]
)

for d in da['@.[banner][:1]']:
    print(d.uri)
```


```text
test-1.jpeg
test-1.jpeg
```

To summarize, slicing can be put in front of the field selector  to restrict the number of dataclass objects; or can be put after the field selector to restrict the number of sub-Documents.

### Select nested fields

A field can be annotated as a DocArray dataclass. In this case, the nested structure from the latter dataclass is copied to the former's `.chunks`. To select the deeply nested field, one can simply follow:

```text
@.[field1, field2, ...].[nested_field1, nested_field1, ...]
```

For example,

```{code-block} python
---
emphasize-lines: 23
---
from docarray import dataclass, Document, DocumentArray
from docarray.typing import Image, Text


@dataclass
class BannerDoc:
    description: Text = 'this is a test empty image'
    banner: Image = 'test-1.jpeg'


@dataclass
class ColumnArticle:
    featured: BannerDoc
    description: Text = 'this is a column article'
    website: str = 'https://jina.ai'


c1 = ColumnArticle(featured=BannerDoc(banner='test-1.jpeg'))
c2 = ColumnArticle(featured=BannerDoc(banner='test-2.jpeg'))

da = DocumentArray([Document(c1), Document(c2)])

for d in da['@.[featured].[banner]']:
    print(d.uri)
```

```text
test-1.jpeg
test-2.jpeg
```