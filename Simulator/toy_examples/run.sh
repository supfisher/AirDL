## Before running this script, you should use the following command
## to apply for a GPU node as master node: "salloc --time 1:00:00 --job-name test --gres=gpu:gtx1080ti:1"
## Then, after a short waiting time, IBex will assign you a gpu node, e.g. gpu502-01,
## All you need: ssh gpu502-01, and use ifconfig to get its ip addr.
## And you need to change the ip:port in the following command for dist_url.

## You can also change the number of world-size, but notice that you should also change
## the world size in run.sh.


sbatch run_on_ibex.sh ## Sometimes it may take a long time to assign a node. Therefore, make sure
#the resources is assigned to you before calling next command

python ./main.py --rank 0 --world_size 2 --dist_url='tcp://10.68.170.167:8091'