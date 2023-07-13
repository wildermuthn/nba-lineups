# Install

- Install [Anaconda](https://www.anaconda.com/products/individual)
- `conda env create`
- `conda activate nba-lineups`
- `pip install pandas nba_api pyarrow wandb`

## Upload Data
- `cd data`
- `tar -zcvf raw.tar.gz raw`
- `gcloud compute scp --recurse --zone us-central1-c /home/wildermuthn/PycharmProjects/nba-lineups/data/raw.tar.gz nba-lineups-v100-1:/home/wildermuthn/nba-lineups/data/raw.tar.gz`

## SSH into VM
- `gcloud compute ssh --zone us-central1-c nba-lineups-v100-1`
- `cd nba-lineups/data`
- `tar -xzvf raw.tar.gz`