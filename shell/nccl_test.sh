export NCCL_SOCKET_IFNAME=eth0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=0  
export NCCL_IB_TIMEOUT=22
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TC=160
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_HCA=mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3,mlx5_bond_4,mlx5_bond_5,mlx5_bond_6,mlx5_bond_7
export NCCL_ALGO=Ring
# pip install -i http://10.68.17.255:8081/repository/pypi/simple --trusted-host 10.68.17.255 athena_train_tools==0.0.59
attreport --nccl_test --master_port=6102 --hoststr="node1 slots=8,node9 slots=8"
