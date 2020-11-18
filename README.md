## Distilling Powerful Student from Online Knowledge Distillation using Feature Fusion and Self-Distillation 

<div align=center><img src="img/framework.png" height = "50%" width = "60%"/></div>

The framework of our proposed FFSD for online knowledge distillation. First, student 1 and student 2 learn from each other in a collaborative way. Then by shifting the attention of student 1 and distilling it to student 2, we are able to enhance the diversity among students. Last, the feature fusion module fuses all the students’ information into a fused feature map. The fused representation is then used to assist the learning of the student leader. After training, we simply adopt the student leader which achieves superior performance over all other students.

### Getting Started

The code has been tested using Pytorch1.5.1 and CUDA10.2 on Ubuntu 18.04.

Please type the command 

```shell
pip install -r requirements.txt
```

to install dependencies.

### FFSD

- You can run the following code to train models on CIFAR-100:

  ```shell
  python cifar.py
  	--dataroot ./database/cifar100
  	--dataset cifar100
  	--model resnet32
  	--lambda_diversity 1e-5
  	--lambda_self_distillation 1000
  	--lambda_fusion 10
  	--gpu_ids 0
  	--name cifar100_resnet32_div1e-5_sd1000_fusion10
  ```

- You can run the following code to train models on ImageNet:

  ```shell
  python distribute_imagenet.py
  	--dataroot ./database/imagenet
  	--dataset imagenet
  	--model resnet18
  	--lambda_diversity 1e-5
  	--lambda_self_distillation 1000
  	--lambda_fusion 10
  	--gpu_ids 0,1
  	--name imagenet_resnet18_div1e-5_sd1000_fusion10
  ```

  

### Experimental Results

We provide the student leader models in the experiments, along with their training loggers and configurations.

|   Model   |  Dataset  | Top1 Accuracy (%) |                           Download                           |
| :-------: | :-------: | :---------------: | :----------------------------------------------------------: |
| ResNet20  | CIFAR-100 |       72.64       | [Link](https://drive.google.com/drive/folders/1FdQK5LQ3e1S0qCJ-FMZkg6J1OfzNJLDm?usp=sharing) |
| ResNet20  | CIFAR-100 |       72.58       | [Link](https://drive.google.com/drive/folders/1N75ZLmAoEqu6zmdIJrQx1h5drvJ7NpnN?usp=sharing) |
| ResNet20  | CIFAR-100 |       72.88       | [Link](https://drive.google.com/drive/folders/1hLxVQcgAwMYDQxjTG7A1Z4n4sIJfscpd?usp=sharing) |
| ResNet32  | CIFAR-100 |       74.92       | [Link](https://drive.google.com/drive/folders/1ahxHYcJMwSWq1iu3fSCa0hmyjW2BYLW_?usp=sharing) |
| ResNet32  | CIFAR-100 |       74.82       | [Link](https://drive.google.com/drive/folders/1SV9572h1TnpiDP8T0wrv0OakltnOBsu8?usp=sharing) |
| ResNet32  | CIFAR-100 |       74.82       | [Link](https://drive.google.com/drive/folders/1T7TaV4lvdvjZS3k3n_2O0eXMJ1pDspIH?usp=sharing) |
| ResNet56  | CIFAR-100 |       75.84       | [Link](https://drive.google.com/drive/folders/1Xrwq1Vc-dmUw6MbOFZoAPgJ3RHOHJ5jg?usp=sharing) |
| ResNet56  | CIFAR-100 |       75.66       | [Link](https://drive.google.com/drive/folders/1wQ2c4rxc8FuspUwG-I0pwP5R40gyX8NV?usp=sharing) |
| ResNet56  | CIFAR-100 |       75.91       | [Link](https://drive.google.com/drive/folders/1gLU6uBQAl24R4lN248dKqZYa8g525Cbq?usp=sharing) |
| WRN-16-2  | CIFAR-100 |       75.87       | [Link](https://drive.google.com/drive/folders/1MF7VLcDjQWAXfe1I39irevrBN0ZgpgNR?usp=sharing) |
| WRN-16-2  | CIFAR-100 |       75.86       | [Link](https://drive.google.com/drive/folders/1FDOfUC1SV59D0vUAsgcBT0Dqctu3Z8Wv?usp=sharing) |
| WRN-16-2  | CIFAR-100 |       75.69       | [Link](https://drive.google.com/drive/folders/1rcxQ_pBF1M5VGuPU_Zla3hl8IVqfsPOv?usp=sharing) |
| WRN-40-2  | CIFAR-100 |       79.13       | [Link](https://drive.google.com/drive/folders/1ECtkuM5-5ZAP4q6Gaszky3SNLWsTMhv5?usp=sharing) |
| WRN-40-2  | CIFAR-100 |       79.19       | [Link](https://drive.google.com/drive/folders/1TSn2oyQuRVteJIwt3nC9p5YUKGxKOJc_?usp=sharing) |
| WRN-40-2  | CIFAR-100 |       79.11       | [Link](https://drive.google.com/drive/folders/1Y92sxigZ4mVMyXoRMjCUvhbo6T2EvMnN?usp=sharing) |
| DenseNet  | CIFAR-100 |       77.29       | [Link](https://drive.google.com/drive/folders/1r4zCULZVXDSz1wZTn9yDa_Vz9ILpivoR?usp=sharing) |
| DenseNet  | CIFAR-100 |       77.70       | [Link](https://drive.google.com/drive/folders/1GWVBY3FtbK_EPULmO_A-8dNIc0DXniUv?usp=sharing) |
| DenseNet  | CIFAR-100 |       77.17       | [Link](https://drive.google.com/drive/folders/19GfMvI78UHuXGYjEZ6P1jWs3aGvxk46k?usp=sharing) |
| GoogLeNet | CIFAR-100 |       81.52       | [Link](https://drive.google.com/drive/folders/1GHKJCrZTqzxZAMSy-s3uGjDY_ee-eRoK?usp=sharing) |
| GoogLeNet | CIFAR-100 |       81.93       | [Link](https://drive.google.com/drive/folders/1yGY8vdiLDsilAHwNF6guMd9Rx-veQTA3?usp=sharing) |
| GoogLeNet | CIFAR-100 |       81.34       | [Link](https://drive.google.com/drive/folders/1QExEqMfsIQaPX3egXp72Teq9DuT0yjnj?usp=sharing) |
| ResNet-18 | ImageNet  |       70.87       | [Link](https://drive.google.com/drive/folders/10METcMuXLC14L48R3NgJahssVgZsAiJT?usp=sharing) |
| ResNet-34 | ImageNet  |       74.69       | [Link](https://drive.google.com/drive/folders/1mFlbgc_EPruHRKgaIS8K3wKDytGS0Keq?usp=sharing) |

You can use the following code to test our models.

```shell
python test.py
	--dataroot ./database/cifar100
	--dataset cifar100
	--model resnet32
	--gpu_ids 0
	--load_path ./resnet32/cifar100_resnet32_div1e-5_sd1000_fusion10_1/modelleader_best.pth
```

### Tips

Any problem, free to contact the authors via emails:[shaojieli@stu.xmu.edu.cn](mailto:shaojieli@stu.xmu.edu.cn).