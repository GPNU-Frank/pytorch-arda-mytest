# pytorch-arda-mytest


## 目录结构
	train.py 训练模型
	test.py 测试模型的准确率
	train_v*.py 和 test_v*.py 试验版本
## 试验内容
	train_v1.py 和 test_v1.py 表示 两个生成器版本
	generator_src 为 MNIST src域 28 * 28 输入样本的生成器
	generator_tgt 为 USPS tat域压缩后 18 * 18 输入样本的生成器
## 试验结果
	train.py :
		红色： generator loss
		蓝色： discriminator loss
		绿色： classfier loss
![train.py](https://raw.githubusercontent.com/GPNU-Frank/pytorch-arda-mytest/master/snapshots/Figure_2.png)

	test.py:
		=== Evaluating models ===
		>>> on source domain <<<
		Avg Loss = 0.06447, Avg Accuracy = 98.18833%
		>>> on target domain <<<
		Avg Loss = 0.43089, Avg Accuracy = 89.43264%


	train_v1.py :
		红色： generator_src loss
		蓝色： discriminator loss
		绿色： classfier loss
![train_v1.py](https://raw.githubusercontent.com/GPNU-Frank/pytorch-arda-mytest/master/snapshots/Figure_2_V1.png)

	test_v1.py
		=== Evaluating models ===
		>>> on source domain <<<
		Avg Loss = 0.03399, Avg Accuracy = 99.19833%
		>>> on target domain <<<
		Avg Loss = 9.57380, Avg Accuracy = 8.56413%



