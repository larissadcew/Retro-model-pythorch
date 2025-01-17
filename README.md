# RETRO Transformer

Uma versão fácil de implementar em PyTorch do [RETRO (Retrieval-Enhanced Transformer)](https://arxiv.org/abs/2112.04426).

## Visão Geral

Esta implementação combina o poder de:
- [labml.ai](https://nn.labml.ai/transformers/retro/index.html) para a arquitetura central do RETRO
- [Hugging Face Accelerate](https://github.com/huggingface/accelerate) para treinamento eficiente em diferentes hardwares (CPU, GPU, TPU)

## Principais Características

- Arquitetura flexível suportando várias configurações de modelo
- Mecanismo de recuperação eficiente para geração de texto aprimorada
- Capacidades de treinamento independentes de hardware
- Integração embutida com BERT para embeddings de fragmentos
- Construção de banco de dados e preparação de conjunto de dados simplificadas

## Início Rápido

Veja como começar com o RETRO:

```python
from retro_transformer.bert import BERTForChunkEmbeddings
from retro_transformer.tools.database import build_database, RetroIndex
from retro_transformer.tools.dataset import build_dataset
from retro_transformer.model import RetroModel, NearestNeighborEncoder
from retro_transformer.tools.train import train

# Configuração
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

# Inicializar embeddings do BERT
bert = BERTForChunkEmbeddings('bert-base-uncased', 'cuda')
index = RetroIndex(config['workspace'], config['chunk_len'], bert=bert)

# Construir banco de dados e conjunto de dados
build_database(config['workspace'], config['text_file'], 
              bert=bert, chunk_len=config['chunk_len'])
num_tokens = build_dataset(config['workspace'], config['text_file'], 
                         chunk_len=config['chunk_len'], index=index)

# Criar codificador
encoder = NearestNeighborEncoder(
    chunk_len=config['chunk_len'],
    n_layers=config['n_layers'],
    d_model=config['d_model'],
    d_ff=config['d_ff'],
    n_heads=config['n_heads'],
    d_k=config['d_k'],
    ca_layers={3}
)

# Inicializar modelo RETRO
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

# Treinar o modelo
train(model, config['workspace'], config['text_file'], 
      chunk_len=config['chunk_len'], d_model=config['d_model'])
```

## Architecture Components

- **BERTForChunkEmbeddings**: Handles text chunk encoding using BERT
- **RetroIndex**: Manages the retrieval database
- **NearestNeighborEncoder**: Processes retrieved neighbor information
- **RetroModel**: Core transformer architecture with retrieval enhancement

## Parâmetros do Modelo

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


