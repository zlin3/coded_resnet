#python resnet/resnet_main.py --train_data_path=cifar10/data_batch* \
#                             --log_root=security4/resnet_model$1 \
#                             --train_dir=security4/resnet_model$1/train \
#                             --dataset='cifar10' \
#                             --num_gpus=1 \
#                             --xor_groups=$1
#export XORS=0
#mkdir -p log/resnet_model$1
#python -u resnet/resnet_main.py --train_data_path=cifar100/train* \
#                             --log_root=log/resnet_model$1 \
#                             --train_dir=log/resnet_model$1/train \
#                             --dataset='cifar100' \
#                             --num_gpus=1 \
#                             --xor_groups=$1 > output.out 2>error.out
#echo $1
python resnet/resnet_main.py --eval_data_path=cifar10/test_batch.bin \
                               --log_root=security4/resnet_model$1 \
                               --eval_dir=security4/resnet_model$1/test \
                               --mode=eval \
                               --dataset='cifar10' \
                               --num_gpus=1 \
                               --xor_groups=$1
