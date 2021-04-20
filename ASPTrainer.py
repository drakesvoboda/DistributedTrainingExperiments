import torch
import torch.distributed as dist

import concurrent.futures
from concurrent.futures import Future

from queue import LifoQueue, Empty, Queue, PriorityQueue

from boilerplate import *

class Reducer():
    def __init__(self, rank: int, world_size: int, model, significance_threshold: float):
        self.significance_threshold = significance_threshold
        self.world_size = world_size
        self.in_flight = []
        self.max_in_flight = 4

        def get_param_dict(module: torch.nn.Module):
            res = {}
            name2key = {}

            idx = 0

            for module_name, module in module.named_modules():
                for param_name, param in module.named_parameters(recurse=False):
                    res[idx] = param
                    name2key[f"{module_name}.{param_name}"] = idx
                    idx += 1

            return res, name2key

        self.model = model
        self.params, self.param_name_to_idx = get_param_dict(model)
        self.lasts = { key: copy.deepcopy(param.data) for key, param in self.params.items() }
        self.current_update = { key: torch.zeros_like(p) for key, p in self.params.items() }

    def reduce(self, param_name):
        with torch.no_grad():
            param_idx = self.param_name_to_idx[param_name]

            #self.current_update[param_idx].add_(self.params[param_idx].grad, alpha=-1e-2)
            #self.params[param_idx].add_(self.params[param_idx].grad, alpha=-1e-2)

            self.current_update[param_idx].add_(self.params[param_idx] - self.lasts[param_idx])
            self.lasts[param_idx].copy_(self.params[param_idx].data)

            significant_mask = (self.current_update[param_idx] / (self.params[param_idx] + 1e-16)) > self.significance_threshold

            if not significant_mask.any():  # if all elements are zeros
                significant_update = torch.sparse.FloatTensor(*self.current_update[param_idx].size())
            else:
                significant_idx = torch.nonzero(significant_mask).t()
                significant_update = torch.sparse_coo_tensor(significant_idx, self.current_update[param_idx][tuple(significant_idx[i] for i in range(significant_idx.shape[0]))], self.current_update[param_idx].size())
                self.current_update[param_idx][tuple(significant_idx[i] for i in range(significant_idx.shape[0]))] = 0

            # My update gets added back after the all reduce
            self.params[param_idx].subtract_(significant_update)

            return dist.all_reduce(significant_update, op=dist.ReduceOp.SUM, async_op=True), significant_update

class ASPTrainer(Trainer):
    def __init__(self, model: torch.nn.Module, criterion: callable, optim_fn: callable, rank: int, world_size: int, significance_threshold: float):
        super().__init__(model, criterion, None)

        torch.distributed.init_process_group(backend='gloo', world_size=world_size, rank=rank, init_method='env://')

        self.model = model
        self.optim_fn = optim_fn

        with torch.no_grad():
            for p in model.parameters():
                dist.broadcast(p.data, 0)

        self.reducer = Reducer(rank, world_size, model, significance_threshold)

        for name, module in self.model.named_modules():
            if len(list(module.children())) > 0: continue
            params = list(module.parameters())
            if len(params) > 0:
                self.first_module = module
                self.first_name = name
                break

        for name, module in model.named_modules():
            if len(list(module.parameters())) <= 0 or len(list(module.children())) > 0: continue
            module.updates = None
            module.optimizer = self.optim_fn(module.parameters(recurse=False))
            module.register_full_backward_hook(P3Trainer.backwards_pass_hook(self.reducer, name))
            module.register_forward_pre_hook(P3Trainer.forward_pre_hook(self.reducer, name))

    @staticmethod
    def backwards_pass_hook(reducer, module_name):
        def hook(self, *args):
            if not self.training: return
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.updates = {param_name: reducer.reduce(f"{module_name}.{param_name}") for param_name, _ in self.named_parameters(recurse=False)}
            self.zero_grad()

        return hook

    @staticmethod
    def forward_pre_hook(reducer, module_name):
        def hook(self, *args):      
            if self.updates == None: return

            with torch.no_grad():
                for param_name, param in self.named_parameters(recurse=False):
                    fut, update = self.updates[param_name]
                    fut.wait()
                    param.data.add_(update / reducer.world_size)
                
                self.updates = None

        return hook

    def training_step(self, input, label):     
        output, loss = self.step(input, label)

        self.model.zero_grad()
        loss.backward()  

        # Backwards hook for the first module in the network is not called by pytorch, call it here manually.
        ASPTrainer.backwards_pass_hook(self.reducer, self.first_name)(self.first_module)
   
        return output, loss   