import torch.multiprocessing as mp
import torch
import torch.distributed as dist
import argparse
import os

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

parser.add_argument('-w', '--wordsize', default=2, type=int)
parser.add_argument('-r', '--rank', default=1, type=int)
parser.add_argument('-b', '--backend', default='gloo', type=str)

def main(args):
    rank = 0
    if args.rank != 0:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()+1


    dist.init_process_group(backend=args.backend, init_method='tcp://10.109.66.97:8001',
                            world_size=args.wordsize, rank=rank)
    print(dist.is_available())
    a = torch.tensor([1, 1, 1, 1, 1, 1, 1])
    b = torch.tensor([1, 1, 1, 1, 1, 1, 1])*2
    a.cuda()
    b.cuda()
    if dist.get_rank()==0:
        dist.send(a, dst=1)
        print(dist.get_rank(), "a: ", a, "b: ", b)
        print(os.system('hostname'))
    elif dist.get_rank()==1:
        dist.recv(b, src=0)
        print(dist.get_rank(), "a: ", a, "b: ", b)
        print(os.system('hostname'))

def test():
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    print('My rank is ', rank)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
    # test()