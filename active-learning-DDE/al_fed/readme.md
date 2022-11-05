## Reproduce the DDE performance
In this step, you can reproduce the DDE performance with the sampling results. You can complete the DDE sampling in "active-learning-DDE/sampling" and move the results into "active-learning-DDE/al_fed/{--dataset}".  
For convenience, we upload our sampling results. You can reproduce the performance by just running the program in "active-learning-DDE/al_fed" and skip other steps.


## Requirement
python >= 3.8  
torch == 1.8.1  
numpy == 1.19.2  
pillow == 8.0.1  





## Scripts
Extract features:
``` 
cd  active-learning-DDE/al_fed
python main.py --dataset=cifar10 --data_dir=../../data/cifar10 --sample_dir=./cifar_10 --gpu=0 --partition_method=homo 
``` 

--data_dir is the path to the location of dataset, --dataset is the name of the dataset (such as cifar10, cifar100, cinic10, etc.), --gpu is the gpu device used, --sample_dir is the path to the sampling results that are used to reproduce performances, --partition_method is the split strategy(homo means iid split, hetero means non-iid split).  

The performances are stored in "active-learning-DDE/al_fed/log" and you can get the accuracy for different sampling rate by open the corresponding file this folder.
