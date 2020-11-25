# HRN
Code for NeurIPS paper: "HRN: A Holistic Approach to One Class Learning"

## Prerequisites
------
Some important packages' versions are as follow:<br>
    scikit-learn == 0.21.3<br>
    torch == 1.2.0<br>

## Usage
------
You can run our code on MNIST directly by this instruction "python3 main.py". 
Meaning of the arguments:

    --max_epochs: the number of epochs of training

    --batch_size: the size of the batches
    
    --lr: the learning rate of the adam optimizer
    
    --n_cpu: the number of cpu threads to use during batch generation
    
    --img_size: the lenth of input image vectors (eg. mnist is 28*28=784)
    
    --num_classes: the number of classes of the dataset
    
    --gpu: choose whether to use gpu
    
    --dataset: choose dataset for experiments

Please cite our paper the code helps you, thanks very much. <br>
@article{hu2020hrn,
  title={HRN: A Holistic Approach to One Class Learning},
  author={Hu, Wenpeng and Wang, Mengyu and Qin, Qi and Ma, Jinwen and Liu, Bing},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
