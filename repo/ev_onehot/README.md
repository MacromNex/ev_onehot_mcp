# EV + Onehot model

From paper [Learning protein fitness models from evolutionary and assay-labelled data](https://www.nature.com/articles/s41587-021-01146-5).

## Install dependencies

### Install the [EVcouplings](https://github.com/debbiemarkslab/EVcouplings):
``` shell
pip install https://github.com/debbiemarkslab/EVcouplings/archive/develop.zip
```

### Install the [plmc package](https://github.com/debbiemarkslab/plmc):
``` shell
cd $HOME (or use another directory for plmc <directory_to_install_plmc> and modify `scripts/plmc.sh` accordingly with the custom directory)
git clone https://github.com/debbiemarkslab/plmc.git
cd plmc
make all-openmp
```
### Index reference fasta with `esl-sfetch` (Optional)
Downloaded UniRef100 in fasta format from [UniProt](https://www.uniprot.org/downloads). 
Index the uniref100 fasta file into ssi with

``` shell
# Download uniref100 database (already done)
wget ftp://ftp.ebi.ac.uk/pub/databases/uniprot/uniref/uniref100/uniref100.fasta.gz
gunzip uniref100.fasta.gz
mkdir data/
mv uniref100.fasta data/

# Index the uniref100 fasta file into ssi with (already done)
esl-sfetch --index data/uniref100.fasta
```
Set the file location of the fasta file in `scripts/jackhmmer.sh`.

## Usage 
### Build the EV model

```shell
bash fitness_model/ev_onehot/build_ev_model.sh example/LanM LanM
```
Inputs: 
- `wt.fasta`: the WT sequence in fasta format.
- `data.csv`: contains two columns: `seq`, `log_fitness`.
  `seq` is the sequence with mutation, and should be the 'same length' as WT seq.
  `log_fitness` is the log enrichment ratio or other log-scale fitness values, where higher is better.

Detail procedure:
```shell
# Run `jackhmmer` alignment using `scripts/jackhmmer.sh`, need wt.fasta
bash scripts/jackhmmer.sh example/LanM 0.5 2  ../data/uniref100/uniref100.fasta
# The outputs will be in `<dataset>/<run_id>`. the final alignment is saved as `alignment.a2m`. 
# The list of full length target sequences is at `target_seqs.fasta` and `target_seqs.txt`.


# Build the EVmutation mkodel using `scripts/plmc.sh`
bash scripts/plmc.sh LanM example/LanM
# The resulting couplings model files can be directly parsed by the `ev+onehot` predictor.
```

## Train the ev+onehot model
```shell
# 5 fold cross validation of the model
bash fitness_model/ev_onehot/run_train.sh example/LanM/ eval 
# or 
python fitness_model/ev_onehot/train.py  example/LanM -cv

# Find best seed split
bash fitness_model/ev_onehot/run_train.sh example/LanM/ seed

# Train final model
bash fitness_model/ev_onehot/run_train.sh example/LanM/ final 20

```

## Reference
Please refere [here](https://github.com/chloechsu/combining-evolutionary-and-assay-labelled-data) for the original readme.
