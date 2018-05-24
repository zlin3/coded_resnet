service ssh start

bash -c "ssh-keygen -t rsa -f ~/.ssh/id_rsa -N ''"
sleep 120
if [ "$M_OR_S" == "master" ]; then
    for (( i = 2; i <= $TOTAL_SLAVES; i++ ))
    do
        bash -c "cat ~/.ssh/id_rsa.pub | sshpass -p 'zhifeng' ssh nerv$i -o StrictHostKeyChecking=no 'mkdir -p ~/.ssh; cat - >> ~/.ssh/authorized_keys'";
    done
fi

#python resnet/resnet_main.py --train_data_path=cifar10/data_batch* \
#                             --log_root=/tmp/resnet_model \
#                             --train_dir=/tmp/resnet_model/train \
#                             --dataset='cifar10' \
#                             --num_gpus=0 \
#                             --xor_groups=$XORS
if [ "$M_OR_S" == "master" ]; then
    mpirun -mca btl ^openib --allow-run-as-root --bind-to none -host localhost,nerv2,nerv3,nerv4,nerv5,nerv6 \
           --mca plm_rsh_no_tree_spawn 1 python -u resnet/resnet_main_parallel_own.py --train_data_path=cifar100/train* \
                                 --log_root=/tmp/resnet_model \
                                 --train_dir=/tmp/resnet_model/train \
                                 --dataset='cifar100' \
                                 --num_gpus=0 \
                                 --xor_groups=$XORS > output.out 2>error.out
fi

while true; do sleep 200; done
