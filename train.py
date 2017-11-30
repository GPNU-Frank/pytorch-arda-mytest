# -*- coding:utf-8 -*-


from models import classifier , discriminator , generator
from misc import params
from misc.utils import save_model
import torch.optim as optim
import torch.nn as nn
from get_data import *
from torch.autograd import Variable
import matplotlib.pyplot as plt
from misc.utils import calc_gradient_penalty

def plot_durations():
	plt.figure(2)
	plt.clf()
	durations_g = torch.FloatTensor(g_loss_durations)
	durations_d = torch.FloatTensor(d_loss_durations)
	durations_c = torch.FloatTensor(c_loss_durations)
	plt.title('Training...')
	plt.xlabel('Episode')
	plt.ylabel('Duration')
	plt.plot(durations_g.numpy(),'r',label="g_loss")
	plt.plot(durations_d.numpy(),'b',label="d_loss")
	plt.plot(durations_c.numpy(),'g',label="c_loss")
	plt.pause(0.001)  # pause a bit so that plots are updated


use_cuda = torch.cuda.is_available()
# 初始化模型
classifier = classifier.Classifier()
critic = discriminator.Discriminator(input_dims=params.d_input_dims,
                                          hidden_dims=params.d_hidden_dims,
                                          output_dims=params.d_output_dims)
generator = generator.Generator()

criterion = nn.CrossEntropyLoss()

optimizer_c = optim.Adam(
				classifier.parameters(),lr = params.learning_rate,betas = (params.beta1,params.beta2)
	)
optimizer_d = optim.Adam(
				critic.parameters(),lr = params.learning_rate,betas = (params.beta1,params.beta2)
	)
optimizer_g = optim.Adam(
				generator.parameters(),lr = params.learning_rate,betas = (params.beta1,params.beta2)
	)
data_itr_src = get_data_iter("MNIST",train=True)
data_itr_tgt = get_data_iter("USPS",train=True)


pos_labels = Variable(torch.Tensor([1]))
neg_lables = Variable(torch.Tensor([-1]))
g_step = 0

g_loss_durations = []
d_loss_durations = []
c_loss_durations = []


# take variable into cuda
if use_cuda:
	generator.cuda()
	critic.cuda()
	classifier.cuda()
	pos_labels = pos_labels.cuda()
	neg_lables = neg_lables.cuda()

#  for 循环
for epoch in range(params.num_epochs):
	#break
	# 训练 鉴别器
	# 开启求 鉴别器的梯度
	for p in critic.parameters():
		p.requires_grad = True
	# 设置 鉴别器的训练步数
	critic_steps = 20

	for step in range(critic_steps):
		# 获得 训练样本
		image_src , label_src = next(data_itr_src)
		image_tgt , _ = next(data_itr_tgt)
		#make_variable
		image_src = Variable(image_src)
		label_src = Variable(label_src.squeeze_())  # squeeze function
		image_tgt = Variable(image_tgt)
		if use_cuda :
			image_src = image_src.cuda()
			label_src = label_src.cuda()
			image_tgt = image_tgt.cuda()

		if image_src.size(0) != params.batch_size or \
						image_tgt.size(0) != params.batch_size:
			continue

		optimizer_d.zero_grad()
		gen_src = generator(image_src)
		cri_src = critic(gen_src.detach()).mean()
		#cri_src.backward(pos_labels)


		gen_tgt = generator(image_tgt).detach()
		cri_tgt = critic(gen_tgt.detach()).mean()


		d_loss = -cri_src + cri_tgt  # 公式4
		d_loss.backward()

		# compute gradient penalty
		gradient_penalty = calc_gradient_penalty(
			critic, gen_src.data, gen_tgt.data)
		gradient_penalty.backward()
		d_loss = d_loss + gradient_penalty
		# optimize weights of discriminator
		#d_loss = - d_loss_src + d_loss_tgt + gradient_penalty
		optimizer_d.step()

		optimizer_d.step()
	#break



	# 训练 分类器
	optimizer_c.zero_grad()
	pred_c = classifier(generator(image_src).detach())
	c_loss = criterion(pred_c,label_src)  # 公式6  求交叉熵
	c_loss.backward()
	optimizer_c.step()


	# 训练 生成器
	# 训练 生成器时  鉴别器的梯度不下降
	for p in critic.parameters():
		p.requires_grad = False

	optimizer_g.zero_grad()

	# 计算 src 分类器的损失
	gen_src = generator(image_src)
	pred_c = classifier(gen_src)
	g_loss_cls = criterion(pred_c,label_src)
	g_loss_cls.backward()

	# 计算 src 鉴别器 对 生成器的损失
	gen_src = generator(image_src)
	g_loss_src = critic(gen_src).mean()


	 
	# # 计算 tgt 鉴别器 对 生成器的损失
	gen_tgt = generator(image_tgt)
	g_loss_tgt = critic(gen_tgt).mean()

	g_loss = g_loss_src - g_loss_tgt # 公式5
	g_loss.backward()


	optimizer_g.step()
	g_step+=1
	#break

	# print info
	if ((epoch + 1) % params.log_step == 0):
		print("Epoch ",epoch +1 , "in " , params.num_epochs ," d_loss :", d_loss.data[0] , " g_loss: ", g_loss.data[0] , " c_loss:",c_loss.data[0])
		d_loss_durations.append(d_loss.data[0])
		g_loss_durations.append(g_loss.data[0])
		c_loss_durations.append(c_loss.data[0])
		plot_durations()

	# save model
	if ((epoch + 1) % params.save_step == 0):
		save_model(critic, "WGAN-GP_critic-{}.pt".format(epoch + 1))
		save_model(classifier,
				   "WGAN-GP_classifier-{}.pt".format(epoch + 1))
		save_model(generator, "WGAN-GP_generator-{}.pt".format(epoch + 1))




plt.ioff()

plt.show()







