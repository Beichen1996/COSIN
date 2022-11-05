## Feature extraction via cross-client contrastive learning encoder
After training the encoder via cross-client contrastive learning, the DDE system sends the trained encoder to each client and the clients extract the feature of their local data. In the following active learning part, these features are not shared among clients due to the privacy regulation. For convenience and speed, we store these features in one file but there is no data exchange at the usage.

We have uploaded the pre-trained model in './pretrain/'. As the weights are large files, please download them from the webpage in https://github.com/Beichen1996/DDE/tree/main/cross-client-contrastive/feature_extraction/pretrain .


## Requirement
python >= 3.8
torch == 1.8.1
numpy == 1.19.2
pillow == 8.0.1


## Scripts
Extract features:
``` 
cd  cross-client-contrastive/feature_extraction
python extract.py --dataset=cifar10 --data_dir='../../data/cifar10' --gpu=0 --pretrained=./pretrain/cifar10_contrastive.pth
``` 

--data_dir is the path to the location of dataset, --dataset is the name of the dataset (such as cifar10, cifar100, cinic10, etc.), --gpu is the gpu device used, --pretrained is the path to the pretrained model, --pretrained should be corresponding to the --data_dir and --dataset. The extracted feature is saved in this folder with {dataset name}+"feature.pt".
