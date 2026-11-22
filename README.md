NEXUS

Building a Language Model from Scratch for Intelligent Website Generation

NEXUS is an experimental language model project focused on building a decoder-only Transformer from scratch, with the long-term goal of converting CV/resume information into structured, personalized websites using HTML and CSS.

Current Implementation

NEXUS currently has an initial decoder-only Transformer implementation built using PyTorch.

Model Architecture

Component

Current Configuration

Architecture

Decoder-only Transformer

Tokenizer

GPT-2 Tokenizer

Vocabulary Size

50,257

Embedding Dimension

256

Attention Heads

4

Transformer Layers

4

Feed Forward Dimension

1,024

Context Length

512

Total Parameters

16,152,320

Weight Initialization

Random

Framework

PyTorch

The GPT-2 tokenizer is reused for tokenization. However, the model's weights are initialized and trained independently without loading pretrained GPT-2 weights.

Training Configuration

Parameter

Initial Configuration

Batch Size

16

Learning Rate

0.0003

Training Steps

2,000

Context Length

512

Training Tokens

20,010

Validation Tokens

1,054

The initial training configuration encountered CUDA out-of-memory errors on an 8 GB GPU. Subsequent experiments achieved a successful training run, but generation quality remains limited.

Current Results

The first working model demonstrated that the Transformer architecture and generation pipeline could produce text.

However, initial generation experiments exposed several limitations:

Generated HTML structures are frequently incomplete.

The model struggles to maintain consistent webpage structure.

Generated content sometimes contains nonsensical text.

The model requires prompting with additional HTML context to produce longer outputs.

Generation quality is insufficient for producing complete websites independently
