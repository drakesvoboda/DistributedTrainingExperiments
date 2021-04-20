import concurrent.futures
from concurrent.futures import Future

import torch
import torch.distributed as dist
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp

from queue import LifoQueue, Empty, Queue, PriorityQueue

from boilerplate import *
from ParameterServer import *

from dataclasses import dataclass, field
from typing import Any

@dataclass(order=True)
class PrioritizedWork:
    priority: int
    item: Any=field(compare=False)

class ReducerThread(threading.Thread):
    def __init__(self, num_trainers: int, num_shards: int, max_in_flight: int, model: torch.nn.Module):
        threading.Thread.__init__(self)
        self.num_trainers = num_trainers
        self.num_shards = num_shards

        self.to_send = PriorityQueue()
        self.in_flight = []
        self.max_in_flight = max_in_flight

        def get_param_dict(module: torch.nn.Module):
            res = {}
            name2key = {}
            priorities = {}

            idx, priority = 0, 0

            for module_name, module in module.named_modules():
                priority += 1
                for param_name, param in module.named_parameters(recurse=False):
                    res[idx] = param
                    priorities[idx] = priority
                    name2key[f"{module_name}.{param_name}"] = idx
                    idx += 1

            return res, name2key, priorities

        def shard_params(params, num_shards=1):
            shards = [{} for _ in range(num_shards)]

            for idx, (key, param) in enumerate(params.items()):
                shards[idx%num_shards][key] = param

            return shards

        self.model = model
        self.params, self.param_name_to_idx, self.priorities = get_param_dict(model)
        shards = shard_params(self.params, self.num_shards)

        self.ps_rref_map = {}

        # Get references to each parameter server shard
        for idx, shard in enumerate(shards):
            param_server_rref = rpc.remote(f"parameter_server_{idx}", get_parameter_server, args=(shard, num_trainers, idx))

            for param_name in shard.keys():
                self.ps_rref_map[param_name] = param_server_rref

        # Sync inital model parameters with parameter server
        with torch.no_grad():
            for key, p in self.params.items():
                fetched = remote_method(ParameterServer.fetch_param, self.ps_rref_map[key], key)
                p.copy_(fetched)

    def reduce(self, param_name) -> Future:
        param_idx = self.param_name_to_idx[param_name]
       
        fut = Future()

        work = PrioritizedWork(self.priorities[param_idx], (param_idx, fut))

        self.to_send.put(work)

        return fut

    def run(self):
        with torch.no_grad():
            while True:
                self.step()

    def step(self):
        self.in_flight = [fut for fut in self.in_flight if not fut.done()]

        if len(self.in_flight) > self.max_in_flight: return

        try:
            send = self.to_send.get()
        except Empty:
            return

        param_idx, fut = send.item

        work = rpc.rpc_async(self.ps_rref_map[param_idx].owner(), ParameterServer.easgd_update, args=(self.ps_rref_map[param_idx], param_idx, self.params[param_idx].data))
        fut.set_result(work)
        self.in_flight.append(work)

class EASGDTrainer(Trainer):
    def __init__(self, model: torch.nn.Module, criterion: callable, optim_fn: callable, rank: int, world_size: int, tau: int, stagger=True):
        super().__init__(model, criterion, None)

        num_trainers = world_size
        num_shards = world_size

        self.ctx = mp.spawn(run_parameter_server, nprocs=1, args=(world_size + rank, world_size * 2, rank), join=False)

        print(f"Trainer {rank} initializing RPC")
        rpc.init_rpc(name=f"trainer_{rank}", rank=rank, world_size=world_size * 2)
        print(f"Trainer {rank} initialized!")

        self.optim_fn = optim_fn

        self.tau = tau

        # Start reducer thread
        self.reducer = ReducerThread(num_trainers, num_shards, 10, model)
        reducer_thread = threading.Thread(target=self.reducer.run)
        reducer_thread.start()

        for name, module in self.model.named_modules():
            if len(list(module.children())) > 0: continue
            params = list(module.parameters())
            if len(params) > 0:
                self.first_module = module
                self.first_name = name
                break

        for idx, (module_name, module) in enumerate(model.named_modules()):
            if len(list(module.parameters())) <= 0 or len(list(module.children())) > 0: continue
            module.updates = None
            module.iteration = idx if stagger else 0
            module.optimizer = self.optim_fn(module.parameters(recurse=False))
            module.register_full_backward_hook(EASGDTrainer.backwards_pass_hook(self.reducer, self.tau, module_name))
            module.register_forward_pre_hook(EASGDTrainer.forward_pre_hook(num_trainers, module_name))

    @staticmethod
    def backwards_pass_hook(reducer, tau, module_name):
        def hook(self, *args):
            if not self.training: return
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.iteration += 1

            if self.iteration % tau == 0: 
                self.updates = {param_name: reducer.reduce(f"{module_name}.{param_name}") for param_name, _ in self.named_parameters(recurse=False)}

        return hook

    @staticmethod
    def forward_pre_hook(num_trainers, module_name):
        def hook(self, *args):         
            if self.updates == None: return

            with torch.no_grad():
                for param_name, param in self.named_parameters(recurse=False):
                    param -= self.updates[param_name].result().wait() * (.9 / num_trainers)
                
                self.updates = None

        return hook

    def train(self, schedule: 'TrainingSchedule'):   
        super().train(schedule)
        rpc.shutdown()
        self.ctx.join()

    def training_step(self, input, label):
        output, loss = self.step(input, label)

        loss.backward()  

        # Backwards hook for the first module in the network is not called by pytorch, call it here manually.
        EASGDTrainer.backwards_pass_hook(self.reducer, self.tau, self.first_name)(self.first_module)
   
        # self.model.zero_grad()
        return output, loss   