(dataclass)=
# Dataclass

```{figure} https://docarray.jina.ai/_images/dataclass-banner.png
:width: 0 %
:scale: 0 %
```

```{figure} img/dataclass-banner.png
:scale: 0 %
```


```{toctree}
:hidden:

construct
access
new-type
example
```


DocArray's dataclass is a high-level API for representing a multimodal document using
{ref}`nested Document structure<recursive-nested-document>`.
It follows the design and idiom of the standard [Python dataclass](https://docs.python.org/3/library/dataclasses.html),
allowing users to represent a complicated multimodal document intuitively and process it easily via DocArray Document/DocumentArray API. 

In a nutshell, DocArray provides a decorator `@dataclass` and a set of multimodal types in `docarray.typing`,
which allows the multimodal document on the left to be represented as the code snippet on the right:

::::{grid} 2
:gutter: 1
:padding: 0

:::{grid-item-card}
:padding: 0

```{figure} img/image-mmdoc-example.png
:width: 100%
```

:::
:::{grid-item-card}
:padding: 0

```python
from docarray import dataclass, Document
from docarray.typing import Image, Text, JSON


@dataclass
class WPArticle:
    banner: Image
    headline: Text
    meta: JSON


a = WPArticle(
    banner='dog-cat-flight.png',
    headline='Everything to know about flying with pets, from picking your seats to keeping your animal calm',
    meta={
        'author': 'Nathan Diller',
        'column': 'By the Way - A Post Travel Destination',
    },
)

doc = Document(a)
```


:::

::::


Under the hood, `doc` is represented as a {class}`~docarray.Document` containing a {property}`~docarray.Document.chunks`
each, for `banner`, `headline` and `meta`.

But the beauty of DocArray's dataclass is that as a user you don't have to reason about `chunks` at all.
Instead, you define your data structure using your own words, and reason in the domain you are most familiar with.

Before we continue, let's first spend some time to understand the problem and the rationale behind this feature.


## What is multi-modality?


```{tip}
It is highly recommended that you first read through the last two chapters on Document and DocumentArray before moving on, as they help you understand the problem we are solving here.
```

A multimodal document is a document that consists of a mixture of data modalities, such as image, text, audio, etc. Let's see some examples in real-world. Considering an article card (left) from The Washington Post and a sound effect card (right) from BBC:


::::{grid} 2

:::{grid-item-card}

```{figure} img/image-mmdoc-example.png
:width: 100%
```

:::
:::{grid-item-card}

```{figure} img/sound-mmdoc-example.png
:width: 100%
```

:::

::::


The left card can be seen as a multimodal document: it consists of a sentence, an image, and some tags (i.e. author, column section). The right one can be seen as a collection of multimodal documents, each of which consists of an audio clip and a sentence description.


In practice, we want to express such multimodal documents via Document and DocumentArray, so that we can process each modality and leverage all DocArray's API, e.g. to embed, search, store and transfer them. That's the purpose of DocArray dataclass. 

## Understanding the problem

Given a multimodal document, we want to represent it via our [Document](../document/index.md) object. What we have learned so far is:
- A Document object is the basic IO unit for almost all [DocArray API](../document/fluent-interface.md).
- Each Document {ref}`can only contain one type <mutual-exclusive>` of data modality.
- A Document can be {ref}`nested<recursive-nested-document>` under `.chunks` or `.matches`.

Having those in mind, to represent a multimodal document it seems that we need to put each modality as a separated Document and then nested them under a parent Document. For example, the article card from The Washington Post would be represented as follows:

::::{grid} 2

:::{grid-item-card}

```{figure} img/image-mmdoc-example.png
:width: 100%
```

:::
:::{grid-item-card}

```{figure} img/mmdoc-example.svg
:width: 100%
```

:::

::::

- `Doc1` the image Document, containing `.uri` of the image and `.tensor` representation of that banner image.
- `Doc2` the text Document, containing `.text` field of the card
- `Doc0` the container Document of `Doc1` and `Doc2`, also contains some meta information such as author name, column name in `.tags`.

Having this representation has many benefits, to name a few:
- One can process and apply deep learning methods on each Document (aka modality) separately.
- Or _jointly_, by leveraging the nested relationship at the parent level.
- One can enjoy all DocArray API, [Jina API](https://github.com/jina-ai/jina), [Hub Executors](https://hub.jina.ai), [CLIP-as-service](https://clip-as-service.jina.ai/) and [Finetuner](https://github.com/jina-ai/finetuner) out of the box, without redesigning the data structure.

## Understanding the challenges

But why do we need a dataclass module, what are the challenges here? 

The first challenge is that such mapping is **arbitrary and implicit**. Given a real-world multimodal document, it is not straightforward to construct such nested structure for new users of DocArray. The example above is simple, so the answer seems trivial. But what if I want to represent the following newspaper article as one Document? 

::::{grid} 2

:::{grid-item-card}

```{figure} img/complicate-example.png
:width: 100%
```

:::
:::{grid-item-card}

```{figure} img/mmdoc-complicated.svg
:width: 100%
```

:::

::::

The second challenge is accessing the nested sub-Document. We want to provide users an easy way to access the nested sub-Document. It should be as easy and consistent as how they construct such Document in the first place.

The final challenge is how to play well with DocArray and Jina Ecosystem, allowing users to leverage existing API, algorithms and models to handle such multimodal documents. To be specific, the user can use multimodal document as the I/O without changing their algorithms and models.   

## What's next

DocArray's dataclass is designed to tackle these challenges by providing an elegant solution based on Python dataclass. It shares the same idiom as Python dataclass, allowing the user to define a multimodal document by adding type annotations. In the next sections, we shall see how it works.

