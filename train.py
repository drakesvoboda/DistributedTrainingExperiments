import argparse
import os
import time

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import RandomSampler

import SVHN as SVHN
import WSJ as WSJ

from boilerplate import *
from ASPTrainer import *
from EASGDTrainer import *
from DDPTrainer import *

# For deterministic runs
torch.manual_seed(0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N', help='number of nodes (default: 1)')
    parser.add_argument('-np', '--num_proc', default=1, type=int, help='number of procs per node')
    parser.add_argument('-nr', '--local_rank', default=0, type=int, help='ranking within the nodes')
    parser.add_argument('--batch_size', default=16, type=int, metavar='N', help='Batch size')
    parser.add_argument('-a', '--address', default="localhost")
    parser.add_argument('-p', '--port', default="9955")
    parser.add_argument('-i', '--iterations', type=int, default=10000)
    parser.add_argument('-th', '--threshold', type=float, default=0.01)
    parser.add_argument('-m', '--model', type=str, default='deep')
    parser.add_argument('-d', '--dataset', type=str, default='SVHN')
    parser.add_argument('-t', '--tau', type=int, default=5)
    parser.add_argument('-tr', '--trainer', type=str, default='DDP')
    args = parser.parse_args()
    args.world_size = args.num_proc * args.nodes

    
    print(args.dataset, args.model, args.trainer)

    # Task 2: Assign IP address and port for master process, i.e. process with rank=0
    os.environ['MASTER_ADDR'] = args.address
    os.environ['MASTER_PORT'] = args.port

    # Spawns one or many processes untied to the first Python process that runs on the file.
    # This is to get around Python's GIL that prevents parallelism within independent threads.
    mp.spawn(train, nprocs=args.num_proc, args=(args,))

def train(proc_num, args):
    rank = args.local_rank * args.num_proc + proc_num   

    if args.dataset == 'SVHN':
        model = SVHN.DeepModel() if args.model == 'deep' else SVHN.ConvNet()
        criterion = F.cross_entropy
        trainset, validset = SVHN.load_datasets('../data/')
        accuracy = SVHN.accuracy
        lr = 1e-2
        collate_fn=None

    elif args.dataset == 'WSJ':
        if args.model == "lstm": 
            model, convert_tokens_to_ids, convert_tags_to_ids, pad_token_id, num_tags = WSJ.make_lstm()
            lr = 1e-3
        elif args.model == "bert": 
            model, convert_tokens_to_ids, convert_tags_to_ids, pad_token_id, num_tags = WSJ.make_bert()
            lr = 1e-4

        criterion = WSJ.TaggerLoss(num_tags)
        trainset, validset, collate_fn = WSJ.load_datasets(convert_tokens_to_ids, convert_tags_to_ids, pad_token_id)
        accuracy = WSJ.TaggerAccuracy

    print(model)    

    num = len(trainset) // args.world_size
    indicies = torch.arange(len(trainset))[num*rank:num*rank+num]
    trainset = torch.utils.data.Subset(trainset, indicies)

    sampler = RandomSampler(trainset, replacement=True, num_samples=args.iterations*args.batch_size)

    trainloader = DataLoader(trainset, args.batch_size, collate_fn=collate_fn, sampler=sampler, num_workers=0)
    validloader = DataLoader(validset, args.batch_size, collate_fn=collate_fn, num_workers=0)

    # model = model.cuda()

    if args.trainer == 'DDP':
        trainer = DDPTrainer(model=model, 
                            criterion=criterion, 
                            optimizer=torch.optim.SGD(model.parameters(), lr=lr), 
                            rank=rank, 
                            world_size=args.world_size)
    elif args.trainer == 'ASP':
        trainer = ASPTrainer(model=model, 
                            criterion=criterion, 
                            optim_fn=lambda params: torch.optim.SGD(params, lr=lr), 
                            rank=rank, 
                            world_size=args.world_size, 
                            significance_threshold=args.threshold)
    elif args.trainer == 'EASGD':
        trainer = EASGDTrainer(model=model, 
                            criterion=criterion, 
                            optim_fn=lambda params: torch.optim.SGD(params, lr=lr), 
                            rank=rank, 
                            world_size=args.world_size, 
                            tau=args.tau,
                            stagger=True)
    elif args.trainer == 'EASGD_0':
        trainer = EASGDTrainer(model=model, 
                            criterion=criterion, 
                            optim_fn=lambda params: torch.optim.SGD(params, lr=lr), 
                            rank=rank, 
                            world_size=args.world_size, 
                            tau=args.tau,
                            stagger=False)

    num_epochs = 10

    callbacks = [
        LogRank(rank),
        Timer(),
        Throughput(),
        TrainingLossLogger(),
        TrainingAccuracyLogger(accuracy),
        Validator(validloader, accuracy),
        Logger()
    ]

    if rank == 0:
        callbacks.append(TensorboardLogger(name=args.trainer, on_epoch_metrics=["Loss/Validation", "Accuracy/Validation", "Throughput (ex/s)"]))

    schedule = TrainingSchedule(trainloader, num_epochs, callbacks)  
    
    start = time.time()

    trainer.train(schedule)

    end = time.time()
    print(end - start, " seconds to train")

if __name__ == '__main__':
    main()
