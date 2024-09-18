HOSTNAME=$(hostname)
if [ $HOSTNAME == "p-phy-dx-a800-node-prod-28" ];
then
NODE_RANK=0
MASTER_ADDR=0.0.0.0
fi

if [ $HOSTNAME == "p-phy-dx-a800-node-prod-38" ];
then
NODE_RANK=1
MASTER_ADDR=10.9.200.93
fi
python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --master_addr=$MASTER_ADDR --master_port=65431 --node_rank=$NODE_RANK test_new_group.py
