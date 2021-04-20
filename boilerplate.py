import time
import datetime 

import torch
from tqdm.auto import tqdm, trange
from torch.utils.tensorboard import SummaryWriter

class TrainModel():
    def __init__(self, model):
        self.model = model
        self.was_training = model.training

    def __enter__(self):
        self.model.train()

    def __exit__(self, type, value, traceback):
        self.model.train(mode=self.was_training)

class EvalModel():
    def __init__(self, model):
        self.model = model
        self.was_training = model.training

    def __enter__(self):
        self.model.eval()

    def __exit__(self, type, value, traceback):
        self.model.train(mode=self.was_training)

class TrainingSchedule():
    """This class is used as a container for a torch.utils.data.DataLoader and a list of training callbacks. 
    The training schedule is passed to the `trainer.train` method.
    Each training callback has access to a shared callback dictionary (`TrainingSchedule.cb_dict`). 
    Callbacks can coordinate by reading and modifying this dictionary. For example, one callback might write a 
    training statistic to the callback dictionary and a second callback might read that statistic and log to the console.
    """
    def __init__(self, dataloader: torch.utils.data.DataLoader, num_epochs: int, callbacks: list, rank: int=0):
        """Initialize a training schedule

        Args:
            dataloader (torch.utils.data.DataLoader): A dataloader used to iterate the training set
            num_epochs (int): Number of epochs to train for
            callbacks (list[TrainCallback]): List of training callbacks
        """
        self.dataloader = dataloader
        self.num_epochs = num_epochs
        self.epoch = 0
        self.iteration = 0
        self.callbacks = callbacks
        self.cb_dict = {}
        self.metrics = []
        self.rank = rank

        for cb in self.callbacks:
            new_metric = cb.register_metric()

            if new_metric is None:
                continue

            if not isinstance(new_metric, list):
                new_metric = [new_metric]

            self.metrics += new_metric

    def data(self):
        for data in tqdm(self.dataloader, desc=f"Epoch {self.epoch+1}", leave=False, position=self.rank):
            self.iteration += 1
            yield data

    def __iter__(self):
        for i in range(self.num_epochs):
            self.epoch = i
            yield i

    def add_callback(self, callback):
        self.callbacks.append(callback)

    def on_train_begin(self, trainer: 'Trainer'):
        for cb in self.callbacks:
            cb.on_train_begin(trainer, self, self.cb_dict)

    def on_epoch_begin(self, trainer: 'Trainer'):
        for cb in self.callbacks:
            cb.on_epoch_begin(trainer, self, self.cb_dict)

    def on_batch_begin(self, trainer: 'Trainer', *args, **kwargs):
        for cb in self.callbacks:
            cb.on_batch_begin(trainer, self, *args, **kwargs)

    def on_batch_end(self, trainer: 'Trainer', loss, output, *args, **kwargs):
        for cb in self.callbacks:
            cb.on_batch_end(trainer, self, self.cb_dict, loss, output, *args, **kwargs)

    def on_epoch_end(self, trainer: 'Trainer'):
        for cb in self.callbacks:
            cb.on_epoch_end(trainer, self, self.cb_dict)

    def on_train_end(self, trainer: 'Trainer'):
        for cb in self.callbacks:
            cb.on_train_end(trainer, self, self.cb_dict)

class Trainer():
    """A Trainer is used to train a pytorch model with support for checkpointing and 
    recording training statistics. Can be overriden for custom training behavior.
    """
    def __init__(self, model: torch.nn.Module, criterion: callable, optimizer: torch.optim.Optimizer):
        """
        Args:
            model (torch.nn.Module): The PyTorch model to train. 
            criterion (callable): A callable function that returns the loss. Should take parameters (output, label) where output is the model's output and label is the label return by the torch.utils.data.Dataset
            optimizer (torch.optim.Optimizer): A PyTorch optimizer. The optimizer should already be initialzied with the model's parameters.
        """
        self.model = model
        self.criterion = criterion    
        self.optimizer = optimizer
        self.running = False
        self.schedule = None

    def forward(self, input):
        """A forward pass through the model

        Args:
            input (torch.tensor): The model's input, must be on the same device as the model.

        Returns:
            torch.tensor: The model's output. Must be on the same device as the input.
        """
        if isinstance(input, dict):
            return self.model(**input)
        else:
            return self.model(input)

    def step(self, input: torch.tensor, label: torch.tensor):
        """A single training step. Can be overridden to define custom behavior. This method will receive the output of a single iteration of the dataloader 
        from the TrainingSchedule.

        Args:
            input (torch.tensor): The model's input
            label (torch.tensor): The label corresponding to the input.

        Returns:
            output (torch.tensor): The model's output given the input. Must be on the same device as the input.
            loss (scalar): The loss computed using `this.criterion`.
        """
        output = self.forward(input)             
        loss = self.criterion(output, label)
        return output, loss       

    def training_step(self, input: torch.tensor, label: torch.tensor):
        output, loss = self.step(input, label)
        self.model.zero_grad()     
        loss.backward()  
        self.optimizer.step()      

        return output, loss            

    def run(self, schedule):
        self.running = True
        self.schedule = schedule

        schedule.on_train_begin(self)       
        
        for _ in self.schedule:
            if not self.running: break
            
            self.schedule.on_epoch_begin(self)
            
            for item in self.schedule.data():
                if not self.running: break
                
                self.schedule.on_batch_begin(self, *item)

                output, loss = self.training_step(*item) 

                self.schedule.on_batch_end(self, loss.data, output, *item)
            
            self.schedule.on_epoch_end(self)      

        self.schedule.on_train_end(self)   

    def train(self, schedule: 'TrainingSchedule'):   
        """Train the trainer's model using a training schedule.

        Args:
            schedule (TrainingSchedule): The training schedule used to train the trainer's model.
        """
        with TrainModel(self.model):
            self.run(schedule)

    def stop(self):
        self.running = False

class TrainCallback:
    """An abstract class representing a training callback. 
    Callback methods are called by the TrainingSchedule class during training. 
    Each callback has access to a shared callback dictionary (`cb_dict`). Callbacks can coordinate 
    by reading and writing to the callback dictionary."""
    def on_train_begin(self, trainer: Trainer, schedule: TrainingSchedule, cb_dict: dict): pass
    def on_epoch_begin(self, trainer: Trainer, schedule: TrainingSchedule, cb_dict: dict): pass
    def on_batch_begin(self, trainer: Trainer, schedule: TrainingSchedule, cb_dict: dict, *args, **kwargs): pass
    def on_batch_end(self, trainer: Trainer, schedule: TrainingSchedule, cb_dict: dict, loss, output, *args, **kwargs): pass
    def on_epoch_end(self, trainer: Trainer, schedule: TrainingSchedule, cb_dict: dict): pass
    def on_train_end(self, trainer: Trainer, schedule: TrainingSchedule, cb_dict: dict): pass
    def register_metric(self): pass

class TrainingAccuracyLogger(TrainCallback):
    """Log's training accuracy to the callback dictionary after each epoch."""
    def __init__(self, accuracy_fn: callable, metric_name = "Accuracy/Train"):
        self.accuracy_fn = accuracy_fn
        self.metric_name = metric_name
        
    def register_metric(self):
        return self.metric_name

    def on_epoch_begin(self, *args, **kwargs):
        self.total = 0
        self.count = 0

    def on_batch_end(self, trainer, schedule, cb_dict, loss, output, input, label):
        self.total += self.accuracy_fn(output, label)
        self.count += 1

    def on_epoch_end(self, trainer, schedule, cb_dict): 
        cb_dict[self.metric_name] = self.total / self.count

class TrainingLossLogger(TrainCallback):
    """Log's training accuracy to the callback dictionary after each epoch."""
    def __init__(self, metric_name = "Loss/Train"):
        self.metric_name = metric_name
        
    def register_metric(self):
        return self.metric_name

    def on_epoch_begin(self, *args, **kwargs):
        self.total = 0
        self.count = 0

    def on_batch_end(self, trainer, schedule, cb_dict, loss, output, input, label):
        cb_dict[self.metric_name] = loss
        self.total += loss
        self.count += 1

    def on_epoch_end(self, trainer, schedule, cb_dict): 
        cb_dict[self.metric_name] = self.total / self.count

class Validator(TrainCallback):
    def __init__(self, dataloader: torch.utils.data.DataLoader, accuracy_fn: callable, metric_names=["Loss/Validation", "Accuracy/Validation"], rank: int = 0):
        self.dataloader = dataloader
        self.accuracy_fn = accuracy_fn
        self.metric_names = metric_names
        self.rank = rank

    def register_metric(self):
        return self.metric_names 

    def run(self, trainer, cb_dict):
        total_loss = 0
        total_acc = 0

        with EvalModel(trainer.model) and torch.no_grad():
            for input, label in tqdm(self.dataloader, desc="Validating", leave=False, position=self.rank):
                output, loss = trainer.step(input, label)
                total_loss += loss.data
                total_acc += self.accuracy_fn(output, label)
        
        cb_dict[self.metric_names[0]] = total_loss / len(self.dataloader)
        cb_dict[self.metric_names[1]] = total_acc / len(self.dataloader)

    def on_epoch_end(self, trainer, schedule, cb_dict, *args, **kwargs): 
        self.run(trainer, cb_dict)

class Timer(TrainCallback):
    def register_metric(self): return 'Wall Time'

    def on_train_begin(self, *args, **kwargs):
        self.start = time.time()

    def on_batch_end(self, trainer, schedule, cb_dict, *args, **kwargs):
        cb_dict['Wall Time'] = time.strftime("%H:%M:%S", time.gmtime(time.time() - self.start))
        
class Logger(TrainCallback):
    """A training callback used to log training statistics to the console. 
    Reads statistics from a TrainingSchedule's callback dictionary (`TrainingSchedule.cb_dict`).
    """
    def __init__(self, metrics: list = None, print_header = True):
        """
        Args:
            metrics (list[str]): List of metrics to log. Each metric is a key used to read from the TrainingSchedule's callback dictionary.
        """
        self.metrics = metrics
        self.printed_header = not print_header

    def divider(self, char="-", sep="+"):
        return sep + sep.join([char * (width+2) for width in self.widths]) + sep

    def format_column(self, columns: list):
        formats = ["{:>" + str(width) + (".4f" if not isinstance(col, (str, int, bool))
                                         else "") + "}" for width, col in zip(self.widths, columns)]
        return ("| " + " | ".join(formats) + " |").format(*columns)

    def on_train_begin(self, trainer, schedule, cb_dict, *args, **kwargs):
        if self.metrics is None:
            self.metrics = []

            for cb in schedule.callbacks:
                new_metric = cb.register_metric()

                if new_metric is None:
                    continue

                if not isinstance(new_metric, list):
                    new_metric = [new_metric]

                self.metrics += new_metric

        self.widths = [len("Epoch")] + [max(7, len(key)) for key in self.metrics]

        cb_dict["print-width"] = sum(self.widths) + (len(self.widths) * 3) + 1

        if not self.printed_header: self.print_header()

    def on_train_end(self, *args, **kwargs):
        tqdm.write(self.divider())

    def print_header(self):
        tqdm.write(self.divider())
        tqdm.write(self.format_column(["Epoch"] + self.metrics))
        tqdm.write(self.divider("="))

    def on_epoch_end(self, trainer, schedule, cb_dict, *args, **kwargs):
        columns = [schedule.epoch+1] + [cb_dict[key] if key in cb_dict and cb_dict[key] is not None else "None" for key in self.metrics]
        metrics_string = self.format_column(columns)

        tqdm.write(metrics_string)

from pathlib import Path

class TensorboardLogger(TrainCallback):
    def __init__(self, directory="./runs/", name='', on_batch_metrics=[], on_epoch_metrics=[]):
        directory = Path(directory)/f"{name}_{time.strftime('%Y%m%d-%H%M%S')}"
        self.writer = SummaryWriter(log_dir=str(directory))
        self.on_batch_metrics = on_batch_metrics
        self.on_epoch_metrics = on_epoch_metrics

    def on_train_begin(self, trainer, schedule, *args, **kwargs):
        if len(self.on_epoch_metrics) == 0:
            self.on_epoch_metrics = schedule.metrics

    def on_batch_end(self, trainer, schedule, cb_dict, loss, *args, **kwargs):
        if self.on_batch_metrics == []:
            self.writer.add_scalar('Loss/train', loss, schedule.iteration)
        else:
            for metric in self.on_batch_metrics:
                if metric in cb_dict and cb_dict[metric] is not None:
                    self.writer.add_scalar(metric, cb_dict[metric], schedule.iteration)

    def on_epoch_end(self, trainer, schedule, cb_dict):
        for metric in self.on_epoch_metrics:
            if metric in cb_dict and cb_dict[metric] is not None:
                self.writer.add_scalar(metric, cb_dict[metric], schedule.iteration)

    def __del__(self):
        self.writer.close()
            
class TorchLRScheduleCallback(TrainCallback):
    def __init__(self, schedule_fn, *args, **kwargs):
        self.schedule_fn = schedule_fn
        self.args = args
        self.kwargs = kwargs

    def on_train_begin(self, trainer, schedule, cb_dict): 
        self.schedule = self.schedule_fn(trainer.optimizer, *self.args, **self.kwargs)

    def register_metric(self): return 'Learning Rate'

    def on_epoch_begin(self, session, schedule, cb_dict): 
        last_lr = self.schedule.get_last_lr()
        if isinstance(last_lr, list): last_lr = last_lr[0]
        cb_dict['Learning Rate'] = "{:.2E}".format(last_lr)

class TorchOnBatchLRScheduleCallback(TorchLRScheduleCallback):    
    def on_batch_end(self, *args, **kwargs):
        self.schedule.step()

class TorchOnEpochLRScheduleCallback(TorchLRScheduleCallback):    
    def on_epoch_end(self, *args, **kwargs):
        self.schedule.step()

class LogRank(TrainCallback):
    def __init__(self, rank):
        self.rank = rank

    def register_metric(self):
        return 'Rank'

    def on_train_begin(self, session, schedule, cb_dict): 
        cb_dict['Rank'] = self.rank

class Throughput(TrainCallback):
    def register_metric(self): return 'Throughput (ex/s)'

    def on_epoch_begin(self, *args, **kwargs):
        self.start = time.time()
        self.count = 0

    def on_epoch_end(self, *args, **kwargs):
        time = time.time - self.start()
        cb_dict['Throughput (ex/s)'] = self.count / time

    def on_batch_end(self, trainer, schedule, cb_dict, loss, output, input, label):
        self.count += input.shape[0]