# DistributedTrainingExperiments
 
 ## Run instructions
 
 This section has instructions for recreating the results reported in my final report for the EECS598 Systems for AI course at Umich.
 
 Everything is run through the `train.py` python script. It has several options to choose the particular model and SGD algorithm.
 
 For example, to train on the SVHN dataset using the MLP model using Staggered EASGD on three nodes, run the following:
 
 ```
 # Node 0
 python train.py -n 3 -a <HEAD NODE ADDRESS> --trainer EASGD --model deep --dataset SVHN -nr 0
 
 # Node 1
 python train.py -n 3 -a <HEAD NODE ADDRESS> --trainer EASGD --model deep --dataset SVHN -nr 1
 
 # Node 2
 python train.py -n 3 -a <HEAD NODE ADDRESS> --trainer EASGD --model deep --dataset SVHN -nr 2
 ```
 
The `--trainer` options are the following:
 
 * `DDP` : Synchronized SGD
 * `EASGD` : Staggered EASGD
 * `EASGD_0` : Standard EASGD
 * `COMPRESS` : Significance Compression

The `--dataset` options are the following:
 
 * `SVHN`
 * `WSJ`
 
The `--model` options are the following:
 
 * `deep` : 7 layer MLP (for use with SVHN)
 * `conv` : 2 layer CNN (for use with SVHN)
 * `lstm` : 1 layer LSTM (for use with WSJ)
 * `bert` : Small transformer (for use with WSJ)


Setting the `--log` flag will log results to tensorboard.
 
