import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
from PIL import Image

# Load the pretrained model
model = models.resnet18(pretrained=True)
# Use the model object to select the desired layer
layer = model._modules.get('avgpool')

def get_vector(image):
	# 2. Create a PyTorch Variable with the transformed image
	t_img = Variable(image).unsqueeze(0)
	# 3. Create a vector of zeros that will hold our feature vector
	#    The 'avgpool' layer has an output size of 512
	my_embedding = torch.zeros(512)
	# 4. Define a function that will copy the output of a layer
	def copy_data(m, i, o):
		my_embedding.copy_(o.data)
	# 5. Attach that function to our selected layer
	h = layer.register_forward_hook(copy_data)
	# 6. Run the model on our transformed image
	model(t_img)
	# 7. Detach our copy function from the layer
	h.remove()
	# 8. Return the feature vector
	return my_embedding

if __name__ == '__main__':
	#Parse Arguments
	parser = argparse.ArgumentParser(description='Convert An Image Folder To a Train CSV by extracting ResNet18 features. Expects a master folder filled with subfolders for each class')
	parser.add_argument('IMGDIR', type=str, help='Master folder name')
	parser.add_argument('CSVFILE', type=str, help='Name of CSV to save to')
	np.random.seed(0)
	args = parser.parse_args()
	#Torchvision Import
	from torchvision import datasets, transforms as T
	#Transforms Required to use ResNet
	transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
	#Load Image Folder
	image_dataset =datasets.ImageFolder(args.IMGDIR,transform)
	#Put Image Dataset into DataLoader
	data_loader = torch.utils.data.DataLoader(image_dataset,batch_size=1)
	#Loop through and create numpy data table
	output_datatable = []
	for image,label in data_loader:
		output_datatable.append(np.concatenate((get_vector(image).numpy().reshape(1, -1), np.array(label).reshape(1,1)), axis=1))
	#Save outputcsv
	np.savetxt(args.CSVFILE, np.array(output_datatable), delimiter=',', fmt='%s')