# test for U-Net (class)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim
from PIL import Image
from torch.autograd import Function
import matplotlib.pyplot as plt
import os
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
import glob
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

### glob.glob()在linux系统下可能会产生乱序的问题，导致训练数据跟mask无法对应，所以加一个排序函数
# get all the image and mask path and number of images
folder_data = sorted(glob.glob("data/train/*.jpg"))
folder_mask = sorted(glob.glob("data/train_masks/*.gif"))


# split these path using a certain percentage
len_data = len(folder_data)
print(len_data)
scale_factor = 0.25 # 0.25
train_size = 0.95# i.e, 0.95, which means 95% of the training data are used for training, 5% for validation
batchsize = 8 # i.e, 8， it depends on your gpu/cpu memory

# Get original image size
height = Image.open(folder_data[0]).size[1]
width = Image.open(folder_data[0]).size[0]
print('original_height: ',height)
print('original_width: ', width)

train_image_paths = folder_data[:int(len_data*train_size)]# extract training data paths
val_image_paths = folder_data[int(len_data*train_size):]#extract validation data paths

train_mask_paths = folder_mask[:int(len_data*train_size)]# extract training mask paths
val_mask_paths = folder_mask[int(len_data*train_size):]# extract validation mask paths



# Customize the class for your own dataset
class CustomDataset(Dataset):
    def __init__(self, image_paths, target_paths, train=True):   # initial logic happens like transform

        self.image_paths = image_paths
        self.target_paths = target_paths
        # for the use of transforms.Compose, please refer to:
        # https://pytorch.org/docs/stable/torchvision/transforms.html
        self.transforms = transforms.Compose([
                            transforms.Resize((int(height*scale_factor), int(width*scale_factor))), # Rescale image size. You can use transforms.Resize() for rescaling
                            transforms.ToTensor() # Convert to torch tensor. You can use transforms.ToTensor()
                            ])
    def __getitem__(self, index):
        # You can use Image.open to load images.
        # Image is a package of PIL. Check https://pillow.readthedocs.io/en/stable/reference/Image.html for more details
        image = Image.open(self.image_paths[index]) 
        mask = Image.open(self.target_paths[index])  # Same for masks
        # read image. You can use Image.open()
#         image = ... 

        # Apply self.transforms to the images
        t_image = self.transforms(image) 
        t_mask = transforms.Resize((int(height*scale_factor),int(width*scale_factor)))(mask)
        t_mask = torch.from_numpy(np.array(t_mask)).unsqueeze(0).float()
        ### 注意此处实验课PPT上的self.transforms(Image.open(mask))是不对的，对mask进行这样的操作会使得所有像素点的label均为0；
        ### 可参考以下写法（将mask进行同样的resize之后手动将转换为tensor）
        ### t_mask =  transforms.Resize((int(height*scale_factor),int(width*scale_factor)))(Image.open(mask)) # Same for masks
        ### t_mask = torch.from_numpy(np.array(t_mask)).unsqueeze(0).float()
        return t_image, t_mask           # Return the rescaled torch-version images and masks

    def __len__(self):  # return count of sample we have

        return len(self.image_paths)

# Create training dataset object with your CustomeDataset
train_dataset = CustomDataset(train_image_paths, train_mask_paths, train=True)

# Same for validation data
val_dataset = CustomDataset(val_image_paths, val_mask_paths, train=False)

# Get the total number of training data
N_train = len(train_dataset)

################################################ [TODO] ###################################################
# DEFINE SINGLE_CONV CLASS

class single_conv(nn.Module):
    '''(conv => BN => ReLU) '''
    def __init__(self, in_ch, out_ch):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x



################################################ [TODO] ###################################################
# DEFINE DOWN CLASS
class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.down = nn.MaxPool2d(2) # use nn.MaxPool2d( )
        self.conv = single_conv(in_ch,out_ch) # use previously defined single_cov
    def forward(self, x):
        x = self.down(x)
        x = self.conv(x)
        return x
    

################################################ [TODO] ###################################################
# DEFINE UP CLASS
# Note that this class will not only upsample x1, but also concatenate up-sampled x1 with x2 to generate the final output

class up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up, self).__init__()       
#         self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) # use nn.Upsample( )
        self.up = nn.Upsample(2)
        self.conv = single_conv(in_ch,out_ch) # use previously defined single_cov

    def forward(self, x1, x2):
        # This part is tricky, so we provide it for you
        # First we upsample x1
        x1 = self.up(x1)
            
        # Notice that x2 and x1 may not have the same spatial size. 
        # This is because when you downsample old_x2(say 25 by 25), you will get x1(12 by 12)   
        # Then you perform upsample to x1, you will get new_x1(24 by 24)
        # You should pad a new row and column so that new_x1 and x2 have the same size.
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        
        # Now we concatenat x2 channels with x1 channels
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

################################################ [TODO] ###################################################
# DEFINE OUTCONV CLASS
class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1) # Use nn.Conv2D( ) since we do not need to do batch norm and relu at this layer

    def forward(self, x):
        x = self.conv(x)
        return x

################################################ [TODO] ###################################################
# Build your network with predefined classes: single_conv, up, down, outconv
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = single_conv(n_channels, 16) # conv2d +  batchnorm + relu
        self.down1 = down(16, 32)         # maxpool2d + conv2d + batchnorm + relu
        self.down2 = down(32, 32)         # maxpool2d + conv2d + batchnorm + relu

        self.up1 = up(64, 16)             # upsample + pad + conv2d + batchnorm + relu
        self.up2 = up(32, 16)             # upsample + pad + conv2d + batchnorm + relu

        self.outc = outconv(16, n_classes) # conv2d

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        x = self.up1(x3, x2)
        x = self.up2(x, x1)

        x = self.outc(x)
        return torch.sigmoid(x)

################################################ [TODO] ###################################################
# define dice coefficient 
class DiceCoeff(Function):
    """Dice coeff for one pair of input image and target image"""
    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001 # in case union = 0
        ################################################ [TODO] ###################################################
        # Calculate intersection and union. 
        # You can convert the input image into a vector with input.contiguous().view(-1)
        # Then use torch.dot(A, B) to calculate the intersection.
        # Use torch.sum(A) to get the sum.
        self.inter = torch.dot(input.contiguous().view(-1), target.contiguous().view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps
        # Calculate DICE
        d = 1. - (2. * self.inter + eps) / self.union
        return d

# Calculate dice coefficients for batches
def dice_coeff(input, target):
    """Dice coeff for batches"""
    s = torch.FloatTensor(1).zero_()
    
    # For each pair of input and target, call DiceCoeff().forward(input, target) to calculate dice coefficient
    # Then average
    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])
    s = s / (i + 1)
    return s

################################################ [TODO] ###################################################
# This function is used to evaluate the network after each epoch of training
# Input: network and validation dataloader
# Output: average dice_coeff
def eval_net(net, dataloader):
    # set net mode to evaluation
    net.eval()
    tot = 0
    for i, b in enumerate(dataloader):
        # Get images and masks from the batch
        img = b[0].to(device)
        true_mask = b[1].to(device)
        ################################################ [TODO] ###################################################      
        # Feed in the image to get predicted mask
        mask_pred = net(img)
        # For all pixels in predicted mask, set them to 1 if larger than 0.5. Otherwise set them to 0
        mask_pred = (mask_pred > 0.5).float()
        # calculate dice_coeff()
        # note that you should add all the dice_coeff in validation/testing dataset together 
        # call dice_coeff() here
        tot += dice_coeff(mask_pred, true_mask).item()
        # Return average dice_coeff()
    return tot / (i + 1)
  
################################################ [TODO] ###################################################
# Create a UNET object. Input channels = 3, output channels = 1
net = UNet(n_channels=3, n_classes=1)
net.to(device)


################################################ [TODO] ###################################################
# Specify number of epochs, image scale factor, batch size and learning rate
epochs = 10 # i.e, 10
lr = 0.001        # i.e, 0.001

# prepare checkpoint dir
dir_checkpoint = 'checkpoints'

################################################ [TODO] ###################################################
# Define an optimizer for your model.
# Pytorch has built-in package called optim. Most commonly used methods are already supported.
# Here we use stochastic gradient descent to optimize
# For usage of SGD, you can read https://pytorch.org/docs/stable/_modules/torch/optim/sgd.html
# Also you can use ADAM as the optimizer
# For usage of ADAM, you can read https://www.programcreek.com/python/example/92667/torch.optim.Adam

optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0.0005)
#OR optimizer = optim.Adam(...)



#suggested parameter settings: momentum=0.9, weight_decay=0.0005

# The loss function we use is binary cross entropy: nn.BCELoss()
criterion = nn.BCELoss()
# note that although we want to use DICE for evaluation, we use BCELoss for training in this example

################################################ [TODO] ###################################################
# Start training
for epoch in range(epochs):
    print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
    net.train()
    ################################################ [TODO] ###################################################
    # Create a dataloader for train_dataset. 
    # Please refer to https://pytorch.org/docs/stable/data.html for the usage of dataloader
    # Make sure you shuffle the dataloader each time you call it
    # For example,
    # torch.utils.data.DataLoader(dataset, batch_size=bsize, shuffle=True, num_workers=2)
    ## 待检查！！！！
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batchsize, shuffle=False, num_workers=0) # Same for valiadation part

    epoch_loss = 0
    
    for i, b in enumerate(train_loader):
        ################################################ [TODO] ###################################################
        # Get images and masks from each batch
        imgs = b[0].to(device)
        true_masks = b[1].to(device)
        ################################################ [TODO] ###################################################
        # Feed your images into the network
        masks_pred = net(imgs)
        # Flatten the predicted masks and true masks. For example, A_flat = A.view(-1)
        masks_probs_flat = masks_pred.view(-1)
        true_masks_flat = true_masks.view(-1)
        ################################################ [TODO] ###################################################
        # Calculate the loss by comparing the predicted masks vector and true masks vector
        # And sum the losses together 
        loss = criterion(masks_probs_flat, true_masks_flat)
        epoch_loss += loss.item()

        if i % 10 == 0: # print loss every 10 step
            print('{0:.4f} --- loss: {1:.6f}'.format(i * batchsize / N_train, loss.item()))

        # optimizer.zero_grad() clears x.grad for every parameter x in the optimizer. 
        # It’s important to call this before loss.backward(), otherwise you’ll accumulate the gradients from multiple passes.
        optimizer.zero_grad()
        # loss.backward() computes dloss/dx for every parameter x which has requires_grad=True. 
        # These are accumulated into x.grad for every parameter x
        # x.grad += dloss/dx
        loss.backward()
        # optimizer.step updates the value of x using the gradient x.grad. 
        # x += -lr * x.grad
        optimizer.step()

    print('Epoch finished ! Loss: {}'.format(epoch_loss / i))
    ################################################ [TODO] ###################################################
    # Perform validation with eval_net()
    val_dice = eval_net(net, val_loader)
    print('Validation Dice Coeff: {}'.format(val_dice))
    # Save the model after each epoch
    torch.save(net.state_dict(),
                dir_checkpoint + 'CP{}.pth'.format(epoch + 1))
    print('Checkpoint {} saved !'.format(epoch + 1))

################################################ [TODO] ###################################################
# Define a function for prediction/testing
def predict_img(net,
                img,
                scale_factor,
                out_threshold):
    '''out_threshold = 0.5'''
    # set the mode of your network to evaluation
    net.eval()
    ################################################ [TODO] ###################################################
    # get the height and width of your image
    img_height = img.size[1]
    img_width = img.size[0]
    
    # resize the image according to the scale factor
    img = img.resize((int(img_width*scale_factor),int(img_height*scale_factor)))
    # Normalize the image by dividing by 255
    img = np.array(img)/255. # (H, W, C)
    # convert from Height*Width*Channel TO Channel*Height*Width
    img = img.transpose((2, 0, 1))
    # convert numpy array to torch tensor 
    X_img = torch.from_numpy(img).unsqueeze(0).float().to(device)
    
    with torch.no_grad():
        ################################################ [TODO] ###################################################
        # predict the masks 
        output_img = net(X_img)
        out_probs = output_img.squeeze(0).cpu()
        print(type(out_probs))
        # Rescale to its original size
        # you can use transforms.ToPILImage(), then transforms.Resize(), transforms.ToTensor()....
        tf = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize((int(height), int(width))), # Rescale image size. You can use transforms.Resize() for rescaling
                        transforms.ToTensor() # Convert to torch tensor. You can use transforms.ToTensor()
                        ])
        out_probs = tf(out_probs)
        # convert to numpy array
        out_mask_np = out_probs.numpy()

    # For all pixels in predicted mask, set them to 1 if larger than 0.5. Otherwise set them to 0
    return out_mask_np > out_threshold

################################################ [TODO] ###################################################
# Load two images from testing dataset
import random
test_folder_data = sorted(glob.glob("data/test/*.jpg"))
test_img_1 = Image.open(random.choice(test_folder_data))
test_img_2 = Image.open(random.choice(test_folder_data))

# Load a model from checkpoint
net.load_state_dict(torch.load('checkpointsCP10.pth'))
# Predict the mask
mask_1 = predict_img(net=net,
                    img=test_img_1,
                    scale_factor=scale_factor,
                    out_threshold=0.5) # you can try different out_threshold values.

mask_2 = predict_img(net=net,
                    img=test_img_2,
                    scale_factor=scale_factor,
                    out_threshold=0.5) # you can try different out_threshold values.

# Plot original images and masks
# If it shows dead kernel, try to uncomment below code

os.environ['KMP_DUPLICATE_LIB_OK']= 'True'

plt.subplot(2,2,1)
plt.imshow(test_img_1)
plt.subplot(2,2,2)
plt.imshow((mask_1.squeeze(0)*255).astype(np.uint8),cmap='gray')
plt.subplot(2,2,3)
plt.imshow(test_img_2)
plt.subplot(2,2,4)
plt.imshow((mask_2.squeeze(0)*255).astype(np.uint8),cmap='gray')
plt.show()