import torch
import torchvision
import torchvision.transforms as transforms
import scipy
import format_cub_dataset_parts as formatter
import cub2011

class BirdLoader(object):
	"""docstring for BirdLoader"""
	def __init__(self, args):
		super(BirdLoader, self).__init__()
		# Change these paths to match the location of the CUB dataset on your machine 
		cub_dataset_dir = "data/CUB_200_2011"
		cub_image_dir = "data/CUB_200_2011/images"

		# Image sizes file
		formatter.create_image_sizes_file(cub_dataset_dir, cub_image_dir)

		# Now we can create the datasets
		train, test, labels = formatter.format_dataset(cub_dataset_dir, cub_image_dir)
		print("MADE IT HERE")

		transform = transforms.Compose(
		    [
		     # TODO: Add data augmentations here

			 #HW 5 Data Augmentations
			 transforms.RandomAffine(30),
			 transforms.RandomHorizontalFlip(),
			 transforms.Resize((128, 128)),
		     transforms.ToTensor(),
		     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		     ])

		transform_test = transforms.Compose([
			transforms.Resize((128, 128)),
		    transforms.ToTensor(),
		    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
		])

		#MAKE A TRAIN SET
		trainset = cub2011.MyCub2011(root='./data', formattedData=train, transform=transform)
		self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchSize,
		                                          shuffle=True, num_workers=2)
								  
		#MAKE A TEST SET								  
		testset = cub2011.MyCub2011(root='./data', formattedData=test, transform=transform_test)
		self.testloader = torch.utils.data.DataLoader(testset, batch_size=args.batchSize,
		                                         shuffle=False, num_workers=2)
	
		# ADD CLASSES
		self.classes = [labels[i] for i in labels]
		


class CifarLoader(object):
	"""docstring for CifarLoader"""
	def __init__(self, args):
		super(CifarLoader, self).__init__()
		transform = transforms.Compose(
		    [
		     # TODO: Add data augmentations here

			 #HW 5 Data Augmentations
			 transforms.RandomAffine(30),
			 transforms.RandomCrop(32, pad_if_needed=True),
		     transforms.ToTensor(),
		     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		     ])

		transform_test = transforms.Compose([
		    transforms.ToTensor(),
		    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
		])

		trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
		                                        download=True, transform=transform)
		self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchSize,
		                                          shuffle=True, num_workers=2)

		testset = torchvision.datasets.CIFAR10(root='./data', train=False,
		                                       download=True, transform=transform_test) 
		self.testloader = torch.utils.data.DataLoader(testset, batch_size=args.batchSize,
		                                         shuffle=False, num_workers=2)

		self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
		
