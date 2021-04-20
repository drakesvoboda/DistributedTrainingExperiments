import threading
import torchvision
import torch
import torch.distributed.rpc as rpc
from torch import optim


def call_method(method, rref, *args, **kwargs):
    return method(rref.local_value(), *args, **kwargs)

def remote_method(method, rref, *args, **kwargs):
    args = [method, rref] + list(args)
    return rpc.rpc_sync(rref.owner(), call_method, args=args, kwargs=kwargs)

class P3ParameterServer(object):
    def __init__(self, params, num_trainers):
        self.num_trainers = num_trainers

        self.iteration = 0

        self.params = params
        self.curr_update_sizes = { key: 0 for key, p in params.items() }
        self.curr_fetch_sizes = { key: 0 for key, p in params.items() }
        #self.optimizers = { key: optim.SGD(p, 1e-2, momentum=.9, weight_decay=0.0001) for key, p in params.items() }
        self.futures = { key: torch.futures.Future() for key, p in params.items() }
        self.locks = { key: threading.Lock() for key, p in params.items() }

        for p in self.params.values():
            p.grad = torch.zeros_like(p)
            p.requires_grad = False

    @staticmethod
    def fetch_param(self, param_id):
        with self.locks[param_id]:
            return self.params[param_id]

    @staticmethod
    def easgd_update(ps_rref, param_id: int, param: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self = ps_rref.local_value()

            with self.locks[param_id]: 
                diff = param - self.params[param_id]
                self.params[param_id] += diff * (.9 / self.num_trainers)

            return diff

    @staticmethod
    @rpc.functions.async_execution
    def reduce_grads(ps_rref, param_id: int, grad: torch.Tensor) -> torch.Tensor:
        # Using the RRef to retrieve the local PS instance
        self = ps_rref.local_value()

        with self.locks[param_id]:        
            self.curr_update_sizes[param_id] += 1
            self.params[param_id].grad += grad

            fut = self.futures[param_id]

            if self.curr_update_sizes[param_id] >= self.num_trainers:
                # update the model
                self.params[param_id].grad /= self.num_trainers
                self.curr_update_sizes[param_id] = 0
                #self.optimizers[param_id].step()
                #self.optimizers[param_id].zero_grad()
                # by setting the result on the Future object, all previous
                # requests expecting this updated model will be notified and
                # the their responses will be sent accordingly.
                fut.set_result(self.params[param_id].grad)

                self.params[param_id].grad = torch.zeros_like(self.params[param_id])
                self.futures[param_id] = torch.futures.Future()

        return fut         

# The global parameter server instance.
param_servers = {}
# A lock to ensure we only have one parameter server.
global_lock = threading.Lock()

def get_parameter_server(params, num_trainers, shard):
    """
    Returns a singleton parameter server to all trainer processes
    """
    global param_servers
    # Ensure that we get only one handle to the ParameterServer.
    with global_lock:
        if shard not in param_servers:
            param_servers[shard] = P3ParameterServer(params, num_trainers)

        return param_servers[shard]

def run_parameter_server(rank, world_size, shard):
    # The parameter server just acts as a host for the model and responds to
    # requests from trainers.
    # rpc.shutdown() will wait for all workers to complete by default, which
    # in this case means that the parameter server will wait for all trainers
    # to complete, and then exit.
    print("PS master initializing RPC")
    rpc.init_rpc(name=f"parameter_server_{shard}", rank=rank, world_size=world_size)
    print(f"RPC initialized! Running parameter server shard {shard}...")
    rpc.shutdown()
    print("RPC shutdown on parameter server.")