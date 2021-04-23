import copy

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
        self.acc = { key: torch.zeros_like(p) for key, p in self.params.items() }
        self.err = { key: torch.zeros_like(p) for key, p in self.params.items() }

    #def reduce(self, param_name):
    #    with torch.no_grad():
    #        param_idx = self.param_name_to_idx[param_name]
    #        self.acc[param_idx] += self.params[param_idx].grad
    #        significant_mask = (self.acc[param_idx].abs() / (self.params[param_idx].abs() + 1e-16)) > self.significance_threshold
    #        if not significant_mask.any():  # if all elements are zeros
    #            sig = torch.sparse.FloatTensor(*self.acc[param_idx].size())
    #        else:
    #            significant_idx = torch.nonzero(significant_mask).t()
    #            sig = torch.sparse_coo_tensor(significant_idx, self.acc[param_idx][tuple(significant_idx[i] for i in range(significant_idx.shape[0]))], self.acc[param_idx].size())
    #            self.acc[param_idx][tuple(significant_idx[i] for i in range(significant_idx.shape[0]))] = 0
    #        return dist.all_reduce(sig, op=dist.ReduceOp.SUM, async_op=True), sig

    def reduce(self, param_name):
        with torch.no_grad():
            param_idx = self.param_name_to_idx[param_name]

            self.acc[param_idx] = self.err[param_idx] + self.params[param_idx].grad
            significant_mask = (self.acc[param_idx].abs() / (self.params[param_idx].abs() + 1e-16)) > self.significance_threshold

            if not significant_mask.any():  # if all elements are zeros
                sig = torch.sparse.FloatTensor(*self.acc[param_idx].size())
            else:
                significant_idx = torch.nonzero(significant_mask).t()
                sig = torch.sparse_coo_tensor(significant_idx, self.acc[param_idx][tuple(significant_idx[i] for i in range(significant_idx.shape[0]))], self.acc[param_idx].size())

            self.err[param_idx] = self.acc[param_idx] - sig

            return dist.all_reduce(sig, op=dist.ReduceOp.SUM, async_op=True), sig

class GradCompressionTrainer(Trainer):
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
            module.register_full_backward_hook(GradCompressionTrainer.backwards_pass_hook(self.reducer, name))
            module.register_forward_pre_hook(GradCompressionTrainer.forward_pre_hook(self.reducer, name))

    @staticmethod
    def backwards_pass_hook(reducer, module_name):
        def hook(self, *args):
            if not self.training: return
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
                    param.grad = update / reducer.world_size

                self.optimizer.step()
                self.optimizer.zero_grad()    
                self.updates = None

        return hook

    def training_step(self, input, label):     
        output, loss = self.step(input, label)

        self.model.zero_grad()
        loss.backward()  

        # Backwards hook for the first module in the network is not called by pytorch, call it here manually.
        GradCompressionTrainer.backwards_pass_hook(self.reducer, self.first_name)(self.first_module)
   
        return output, loss   