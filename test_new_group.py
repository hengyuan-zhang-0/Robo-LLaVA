import torch

torch.distributed.init_process_group(backend="nccl")
cpu_comm = torch.distributed.new_group(backend="gloo")
