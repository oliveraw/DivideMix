# DivideMix applied to USGS dataset

First, install the necessary packages:
```bash
pip install -r requirements.txt
```

Take a look at the scripts found in `scripts/` to run various experiments. For example, `scripts/run_usgs_dividemix_sym0.3_CE.sh` will run the DivideMix algorithm on the USGS dataset, with 30% symmetric noise, trained with standard cross entropy loss. 

The main training code can be found in `Train_usgs.py` and the dataloader code can be found in `dataloader_usgs.py`. Each training run will generate 2 model checkpoints, TSNE and GMM visualizations, and a confusion matrix for each run over the test set. 

To run the "no DivideMix" version of the code, simply set the warmup epochs to be >= the total epochs. This will train two standard classifiers without using the DivideMix algorithm, and at test time both model's logits will be added together for the final prediction. This was done for ease of implementation, as the warmup is already implemented as separate training of the two classifiers without using dividemix. 

### Interactive TSNE
At the end of each run, a json file containing the TSNE coordinates and image paths is created in the <output_dir>/epoch_200 directory. To use this in the interactive visualization, you need to copy this json file into the `.../web_visualization/data_json` directory. 

Additionally, you need to update the following line in `.../web_visualization/app/page.js` to use the new json file for the webpage.
```
'use client'

import Image from "next/image";
import { useState } from "react";

import data_json from "@/data_jsons/nodividemix-sym0.3.json"    // this string must have the filename changed to the filename of the json file you just copied over
import ScatterPlot from "@/components/customChart/ScatterPlot";
```
