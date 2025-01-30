# Analysis of static and dynamic batching algorithms for graph neural networks
Message Passing Graph Neural Network with Edge Updates in Jraph.

This code implements several GNN models. It implements the [SchNet](https://arxiv.org/pdf/1712.06113) model. It implements a message passing model with edge updates, MPEU, graph neural network with the architecture described in this [article](https://arxiv.org/pdf/1806.03146.pdf). There is also the option to run the [PaiNN](https://proceedings.mlr.press/v139/schutt21a.html) model.

## Python library requirements:  
See requirements.txt.

The library uses Jraph to build the GNNs, a library written by DeepMind, which has support for JAX.

## Datasets 
At the moment this can be run with two different datasets: QM9 and aflow.

The QM9 dataset is pulled from spektral and converted into graphs. To get the QM9 dataset run datahandler_QM9.py.

The aflow dataset pulled directly with the alfow API with a json response. To do this first run datapuller.py, you may have to make a directory "aflow". Then run datahandler.py to convert the raw data from the pull into graphs.

## Training
To train a model run main.py, this defaults to the QM9 dataset, but does not automatically pull it. You have to specify config directory, where parameters are pulled and working directory, where results are stored.

The config directory can be one of the two files in configs.

## Timing measurements
The timing measurements are contained in the train.py file.

## Batch statistics
The statistics about the graphs in the batch before and after padding can be found in the stats_before_padding branch. The plotting scripts for the plots in the paper can also be found in this branch.



