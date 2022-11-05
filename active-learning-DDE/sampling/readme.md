## Feature extraction via cross-client contrastive learning encoder
In this stage, DDE selects the most representative samples for AL sampling.

## Requirement
python >= 3.8  
torch == 1.8.1  
numpy == 1.19.2  
pillow == 8.0.1  


## Scripts
Train the DDE and complete AL sampling:
``` 
cd  active-learning-DDE/sampling
python main.py --dataset=cifar10 --num=50000 --data_dir=../../data/cifar10 --gpu=0 --sample_dir=../random_init/cifar_10  --partition_method=homo
``` 

--data_dir is the path to the location of dataset, --dataset is the name of the dataset (such as cifar10, cifar100, cinic10, etc.), --gpu is the gpu device used, --sample_dir are the locations of data split file and initial labeled file, --partition_method is the split strategy(homo means iid split, hetero means non-iid split). In --sample_dir we complete our initialization code with strict iid or non-iid split.

The sampling results are stored in 'active-learning-DDE/sampling/{--dataset}'.
