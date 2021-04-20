import torch
import torch.distributed as dist

from boilerplate import *

class DDPTrainer(Trainer):
    def __init__(self, model: torch.nn.Module, criterion: callable, optimizer, rank: int, world_size: int):
        torch.distributed.init_process_group(backend='gloo', world_size=world_size, rank=rank, init_method='env://') 
        model = torch.nn.parallel.DistributedDataParallel(model)
        super().__init__(model, criterion, None)