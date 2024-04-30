# DivideMix applied to USGS dataset

First, install the necessary packages:
```bash
pip install -r requirements.txt
```

Take a look at the scripts found in `scripts/` to run various experiments. For example, `scripts/run_usgs_dividemix_sym0.3_CE.sh` will run the DivideMix algorithm on the USGS dataset, with 30% symmetric noise, trained with standard cross entropy loss. 

The main training code can be found in `Train_usgs.py` and the dataloader code can be found in `dataloader_usgs.py`. Each training run will generate 2 model checkpoints, TSNE and GMM visualizations, and a confusion matrix for each run over the test set. 
