## Vision Transformer

[Vision Transformer](https://arxiv.org/abs/2010.11929) is a transformer encoder model (BERT-like) pretrained on a large collection of images in a supervised fashion, namely ImageNet-21k, at a resolution of 224x224 pixels. Images are presented to the model as a sequence of fixed-size patches (resolution 16x16), which are linearly embedded. One also adds a [CLS] token to the beginning of a sequence to use it for classification tasks. One also adds absolute position embeddings before feeding the sequence to the layers of the Transformer encoder.

By pre-training the model, it learns an inner representation of images that can then be used to extract features useful for downstream tasks: if you have a dataset of labeled images for instance, you can train a standard classifier by placing a linear layer on top of the pre-trained encoder. One typically places a linear layer on top of the [CLS] token, as the last hidden state of this token can be seen as a representation of an entire image.
</br>


#### Motivation

In order to apply the transformer used in NLP tasks to computer vision, the transformer architecture is minimally modified to process images. And After splitting the image into patched, linearly transform it and use it as input to the transformer.
</br>

#### Architecture

<p align="center"><img width="100%" src="png/ViT.png" /></p>