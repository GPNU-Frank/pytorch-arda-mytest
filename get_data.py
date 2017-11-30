

import torch
from torchvision import datasets, transforms
import datasets.usps as  usps
import datasets.mnist as mnist
from PIL import Image,ImageDraw,ImageFont,ImageFilter
import numpy as np
import matplotlib.pyplot as plt
import  cv2

 # options  : dir  name   batch_size
# dataset_mean = (dataset_mean_value, dataset_mean_value, dataset_mean_value) 0.5
# dataset_std = (dataset_std_value, dataset_std_value, dataset_std_value) 0.5
# def get_mnist(options , train = True ):
# 	#pre_process = transforms.Compose([transforms.ToTensor()])
# 	# pre_process = transforms.Compose([transforms.ToTensor(),
#  #                                       transforms.Normalize(
#  #                                           mean=options['dataset_mean'],
#  #                                           std=options['dataset_std'])])
# 	mnist_dataset = datasets.MNIST(root=options['dir'],
#                                    train=train,
#                                    transform=transforms.ToTensor(),
#                                    download=True)
# 	mnist_data_loader = torch.utils.data.DataLoader(
#         dataset=mnist_dataset,
#         batch_size=options['batch_size'],
#         shuffle=True)
# 	return mnist_data_loader

def get_data_iter(name ,train):
	if name == 'MNIST':
		return get_inf_iterator( mnist.get_mnist(train = True))
	else:
		return get_inf_iterator( usps.get_usps(train = True))


def get_data_loader(name, train=True):
    """Get data loader by name."""
    if name == "MNIST":
        return mnist.get_mnist(train)
    elif name == "USPS":
        return usps.get_usps(train)

def get_inf_iterator(data_loader):
    """Inf data iterator."""
    while True:
        for images, labels in data_loader:
            yield (images, labels)

def make_lower_size(image_tgt):
	lowwer_image_tgt = []
	for image in image_tgt:
		new_data = image.numpy().reshape(28, 28) * 255
		new_im = Image.fromarray(new_data.astype(np.uint8))
		#new_im.show()
		new_im = new_im.resize((18, 18), Image.ANTIALIAS)
		new_mat = np.matrix(new_im.getdata())
		new_mat = (new_mat / 255).reshape(18,18)
		new_mat = new_mat[np.newaxis,:,:]
		lowwer_image_tgt.append(new_mat)
	lowwer_image_tgt = torch.FloatTensor(np.array(lowwer_image_tgt))
	return lowwer_image_tgt

def make_larger_size(image_tgt):
	lowwer_image_tgt = []
	for image in image_tgt:
		new_data = image.numpy().reshape(28, 28) * 255
		new_im = Image.fromarray(new_data.astype(np.uint8))
		#new_im.show()
		new_im = new_im.resize((36, 36), Image.ANTIALIAS)
		new_mat = np.matrix(new_im.getdata())
		new_mat = (new_mat / 255).reshape(36,36)
		new_mat = new_mat[np.newaxis,:,:]
		lowwer_image_tgt.append(new_mat)
	lowwer_image_tgt = torch.FloatTensor(np.array(lowwer_image_tgt))
	return lowwer_image_tgt

if __name__ == '__main__':
	data_itr_tgt = get_data_iter("USPS", train=True)
	image_tgt, label_tgt = next(data_itr_tgt)
	image_tgt = image_tgt[0:2]
	print(image_tgt.shape)
	print(image_tgt)
	new_tgt = make_larger_size(image_tgt)
	print(new_tgt.shape)
	print(new_tgt)
	plt.imshow(new_tgt[0].numpy().reshape(36, 36), cmap="gray")
	plt.show()
	print(label_tgt[0])
	exit(0)
	#options = { 'dir' : 'data' , 'name' : 'MNIST' , 'batch_size' : 64 , 'dataset_mean' : (0.5,0.5,0.5) , 'dataset_std' : (0.5,0.5,0.5)}
	mnist_loader = mnist.get_mnist(train = True)
	usps_loader = usps.get_usps(train = True)
	im = Image.open("snapshots/Figure_2.png")
	im = im.convert("L")
	#im = im.resize((image_width, image_height))
	# im.show()
	data = im.getdata()
	data = np.matrix(data)
	print(data.shape)
	for tgt_img , tgt_label in usps_loader:
		#print(type(tgt_img))
		#print(tgt_img[0])
		plt.imshow(tgt_img[0].numpy().reshape(28,28),cmap="gray")

		plt.show()
		print(tgt_img[0].numpy())
		new_data = tgt_img[0].numpy().reshape(28,28) * 255
		new_im = Image.fromarray(new_data.astype(np.uint8))
		new_im.show()
		#new_im.convert("L")
		print(new_im.size)
		new_im = new_im.resize((18,18), Image.ANTIALIAS)
		new_mat = np.matrix(new_im.getdata())
		plt.imshow(new_mat.reshape(18, 18), cmap="gray")
		plt.show()
		print(new_mat / 255)
		print(tgt_label[0])
		#cv2.imshow("test",tgt_img[0].transpose(0,2).numpy())
		#print(tgt_img[0].size)
		break
	# print(type(mnist_loader))
	# for img , label in mnist_loader:
	# 	print(type(img))
    #
	# 	print(type(img[0]))
	# 	print(img[0])
	# 	print(type(label))
	# 	print(type(label[0]))
	# 	print(label[0])
	# 	break
	#print (type(x))



#from misc import params


# def get_mnist(train):
#     """Get MNIST dataset loader."""
#     # image pre-processing
#     pre_process = transforms.Compose([transforms.ToTensor(),
#                                       transforms.Normalize(
#                                           mean=params.dataset_mean,
#                                           std=params.dataset_std)])

#     # dataset and data loader
#     mnist_dataset = datasets.MNIST(root=params.data_root,
#                                    train=train,
#                                    transform=pre_process,
#                                    download=True)

#     mnist_data_loader = torch.utils.data.DataLoader(
#         dataset=mnist_dataset,
#         batch_size=params.batch_size,
#         shuffle=True)

#     return mnist_data_loader
