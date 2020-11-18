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

#### FFSD

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
| ResNet20  | CIFAR-100 |       72.64       | [Link](https://drive.google.com/file/d/1juofKZh6si3_oBNUKGRLatRFgcz4g6GL/view?usp=sharing) |
| ResNet20  | CIFAR-100 |       72.58       | [Link](https://drive.google.com/file/d/1yyU5QeAj9I3lDe_e5ZXiEa2uJIyIFyBN/view?usp=sharing) |
| ResNet20  | CIFAR-100 |       72.88       | [Link](https://drive.google.com/file/d/1juofKZh6si3_oBNUKGRLatRFgcz4g6GL/view?usp=sharing) |
| ResNet32  | CIFAR-100 |       74.92       | [Link](https://drive.google.com/file/d/1EOtnr1viTWZxnemh7PxNQYp9k6qBBydk/view?usp=sharing) |
| ResNet32  | CIFAR-100 |       74.82       | [Link](https://drive.google.com/file/d/1EOtnr1viTWZxnemh7PxNQYp9k6qBBydk/view?usp=sharing) |
| ResNet32  | CIFAR-100 |       74.82       | [Link](https://drive.google.com/file/d/1o9DqyxrXTHVviAKAq0dkRwZoicbYZsPg/view?usp=sharing) |
| ResNet56  | CIFAR-100 |       75.84       | [Link](https://drive.google.com/file/d/1eDQJZXS2Vctt_8uPr1tZfOZOvsZFO-1x/view?usp=sharing) |
| ResNet56  | CIFAR-100 |       75.66       | [Link](https://drive.google.com/file/d/1Yh7Hn28cEk8A19HaH_hALpskmulREvPF/view?usp=sharing) |
| ResNet56  | CIFAR-100 |       75.91       | [Link](https://drive.google.com/file/d/1g-GtEc_ev8yjpyjeqqDMgqnRKOYF3r3M/view?usp=sharing) |
| WRN-16-2  | CIFAR-100 |       75.87       |                                                              |
| WRN-16-2  | CIFAR-100 |       75.86       |                                                              |
| WRN-16-2  | CIFAR-100 |       75.69       |                                                              |
| WRN-40-2  | CIFAR-100 |       79.13       |                                                              |
| WRN-40-2  | CIFAR-100 |       79.19       |                                                              |
| WRN-40-2  | CIFAR-100 |       79.11       |                                                              |
| DenseNet  | CIFAR-100 |       77.29       |                                                              |
| DenseNet  | CIFAR-100 |       77.70       |                                                              |
| DenseNet  | CIFAR-100 |       77.17       |                                                              |
| GoogLeNet | CIFAR-100 |       81.52       |                                                              |
| GoogLeNet | CIFAR-100 |       81.93       |                                                              |
| GoogLeNet | CIFAR-100 |       81.34       |                                                              |
| ResNet-18 | ImageNet  |       70.87       |                                                              |
| ResNet-34 | ImageNet  |       74.69       |                                                              |

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