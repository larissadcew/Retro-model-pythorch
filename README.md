# RETRO Transformer

An easy-to-implement PyTorch version of [RETRO (Retrieval-Enhanced Transformer)](https://arxiv.org/abs/2112.04426).

## Overview

This implementation combines the power of:
- [labml.ai](https://nn.labml.ai/transformers/retro/index.html) for core RETRO architecture
- [Hugging Face Accelerate](https://github.com/huggingface/accelerate) for efficient training across different hardware (CPU, GPU, TPU)

## Key Features

- Flexible architecture supporting various model configurations
- Efficient retrieval mechanism for enhanced text generation
- Hardware-agnostic training capabilities
- Built-in BERT integration for chunk embeddings
- Streamlined database construction and dataset preparation

## Quick Start

Here's how to get started with RETRO:

```python
from retro_transformer.bert import BERTForChunkEmbeddings
from retro_transformer.tools.database import build_database, RetroIndex
from retro_transformer.tools.dataset import build_dataset
from retro_transformer.model import RetroModel, NearestNeighborEncoder
from retro_transformer.tools.train import train

# Configuration
config = {
    'chunk_len': 16,
    'd_model': 128,
    'd_ff': 512,
    'n_heads': 16,
    'd_k': 16,
    'n_layers': 16,
    'workspace': './workspace',
    'text_file': 'text.txt'
}

# Initialize BERT embeddings
bert = BERTForChunkEmbeddings('bert-base-uncased', 'cuda')
index = RetroIndex(config['workspace'], config['chunk_len'], bert=bert)

# Build database and dataset
build_database(config['workspace'], config['text_file'], 
              bert=bert, chunk_len=config['chunk_len'])
num_tokens = build_dataset(config['workspace'], config['text_file'], 
                         chunk_len=config['chunk_len'], index=index)

# Create encoder
encoder = NearestNeighborEncoder(
    chunk_len=config['chunk_len'],
    n_layers=config['n_layers'],
    d_model=config['d_model'],
    d_ff=config['d_ff'],
    n_heads=config['n_heads'],
    d_k=config['d_k'],
    ca_layers={3}
)

# Initialize RETRO model
model = RetroModel(
    n_vocab=num_tokens,
    d_model=config['d_model'],
    n_layers=config['n_layers'],
    chunk_len=config['chunk_len'],
    n_heads=config['n_heads'],
    d_k=config['d_k'],
    d_ff=config['d_ff'],
    encoder=encoder,
    ca_layers={3, 5}
)

# Train the model
train(model, config['workspace'], config['text_file'], 
      chunk_len=config['chunk_len'], d_model=config['d_model'])
```

## Architecture Components

- **BERTForChunkEmbeddings**: Handles text chunk encoding using BERT
- **RetroIndex**: Manages the retrieval database
- **NearestNeighborEncoder**: Processes retrieved neighbor information
- **RetroModel**: Core transformer architecture with retrieval enhancement

## Model Parameters

- `chunk_len`: Length of text chunks for retrieval
- `d_model`: Model dimension
- `d_ff`: Feed-forward network dimension
- `n_heads`: Number of attention heads
- `d_k`: Dimension of keys in attention mechanism
- `n_layers`: Number of transformer layers
- `ca_layers`: Cross-attention layer positions

## Contributing

Feel free to open issues and pull requests to help improve this implementation!

## Citation

If you use this implementation in your research, please cite the original RETRO paper:

```bibtex
@article{borgeaud2021improving,
  title={Improving language models by retrieving from trillions of tokens},
  author={Borgeaud, Sebastian and Mensch, Arthur and Hoffmann, Jordan and Cai, Trevor and Rutherford, Eliza and Millican, Katie and ...},
  journal={arXiv preprint arXiv:2112.04426},
  year={2021}
}
```

## License

This project is open-source and available under the MIT License.
