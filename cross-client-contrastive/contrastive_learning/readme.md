## Cross-client contrastive learning



## Requirement
python >= 3.8
torch == 1.8.1
numpy == 1.19.2
pillow == 8.0.1


## Experiment Scripts
Train the contrastive learning encoder:
``` 
python cross_client_CL.py --dataset=cifar10 --data_dir='../../data/cifar10' --partition_method=homo --comm_round=1000 --gpu=0
``` 

--dataset is the name of the dataset, --data_dir is the path to the location of dataset, --comm_round is the training epoch for contrastive learning and partition_method is the IID(homo) or non-IID(heter) setting. Trained encoder is saved into local folder "./{dataset}".


## For reimplement
As contrastive learning spends a large amount of running time(maybe few days), you can implement this method with our pretrained model weight to skip this contastive learning training stage.
The pretrained model weight is located in the "../feature_extraction" folder and you can extract the DDE feature with the pretrained contrastive encoder at that folder. As the weights are large files, please download them from the webpage in https://github.com/Beichen1996/DDE/tree/main/cross-client-contrastive/feature_extraction/pretrain .
