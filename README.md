# GFR-IL
Generative Feature Replay For Class-Incremental Learning

## Requirements
Python3  and Pytorch 1.1


## CIFAR-100 
num_task: 5 / 10 / 25
- `python train.py -data cifar100 -num_task 10 -epochs_gan 501 -tradeoff 1 -epochs 201 -lr_decay_step 200 -log_dir cifar100_10tasks -dir /dataset -gpu 0`
- `python test.py -data cifar100 -num_task 10 -epochs 201  -dir /dataset -gpu 0 -r checkpoints/cifar100_10  -name cifar100_10`

## ImageNet_sub
num_task: 5 / 10 / 25
- `python train.py -data imagenet_sub -num_task 10 -epochs_gan 201 -tradeoff 1 -epochs 101 -lr_decay_step 100 -log_dir imagenet_sub_10tasks -dir /dataset -gpu 0`
- `python test.py -data imagenet_sub -num_task 10 -epochs 101  -dir /dataset -gpu 0 -r checkpoints/imagenet_sub_10  -name imagenet_sub_10`

## ImageNet_full
num_task: 5 / 10 / 25
- `python train.py -data imagenet_full -num_task 10 -epochs_gan 201 -tradeoff 1 -epochs 101 -lr_decay_step 100 -log_dir imagenet_full_10tasks -dir /dataset -gpu 0 -num_class 1000 -nb_cl_fg 500`
- `python test.py -data imagenet_full -num_task 10 -epochs 101  -dir /dataset -gpu 0 -r checkpoints/imagenet_full_10  -name imagenet_full_10 -num_class 1000 -nb_cl_fg 500 `
