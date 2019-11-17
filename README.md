# unidirectional-NMT

### Set-up the environment

### Get the data

### Shuffle and split the data
See `data/README.md` for instructions

### Preprocess the data
onmt_preprocess --config config/config-preprocess

### Process word embeddings
OpenNMT-py/tools/embeddings_to_torch.py -emb_file_enc "data/fasttext/cc.en.300.vec" -emb_file_dec "data/fasttext/cc.fr.300.vec" -dict_file "data/data.vocab.pt" -output_file "data/embeddings"

### Train the model
onmt_train --config config/config-train.yml
