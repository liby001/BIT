import xc
import stack as st
import decoder as dc
import torch.utils.model_zoo as model_zoo
import torch
import torch.optim as optim
import torch.nn.functional as F
import imageio
LR = 0.001
num_classes=128

encoder=xc.xception()
encoder.cuda()
gan=st.G_NET()
gan.cuda()
decoder=dc.Decoder(num_classes)
decoder.cuda()
#from torchvision import datasets,transforms
import numpy as np
from skimage import transform

LR = 0.001
epoch=1
optimizer =  optim.Adam(gan.parameters(), lr=LR)
state = {'net': gan.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
pretrained_settings = {
        'xception': {
            'imagenet': {
                'url': 'http://data.lip6.fr/cadene/pretrainedmodels/xception-43020ad28.pth',
                'input_space': 'RGB',
                'input_size': [3, 128, 256],
                'input_range': [0, 1],
                'mean': [0.5, 0.5, 0.5],
                'std': [0.5, 0.5, 0.5],
                'num_classes': 1000,
                'scale': 0.8975 # The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
            }
        }
    }
def data(i,batch_size=2):
    input_images = []
    label_images = []
    for j in range(batch_size):
        img1=imageio.imread("F:/datasets/cityscapes/train/train_images/%d.png" %(i%1000+2))
        img1 = transform.resize(img1, (128, 256))
        input_images.append(np.expand_dims(img1, axis=0))

        img2=imageio.imread("F:/datasets/cityscapes/train/train_labels/%d.png" %(i%1000+2))
        img2= transform.resize(img2, (128, 256))
        label_images.append(np.expand_dims(img2,axis=0))
        i=i%2976+1
    input_images = np.squeeze(np.stack(input_images, axis=1))
    label_images = np.squeeze(np.stack(label_images, axis=1))
    input_images = np.reshape(input_images, (-1, 3, 128, 256))
    label_images = np.reshape(label_images, (-1, 4, 128, 256))
    images = torch.from_numpy(input_images)
    images = images.type(torch.FloatTensor)
    images = images.cuda()
    labels = torch.from_numpy(label_images)
    labels = labels.type(torch.FloatTensor)
    labels = labels.cuda()
    return images,labels,i
def pretrain(pre_encoder=True,pre_gan=True):

    pretrain=pre_encoder
    if pretrain:
        pretrained='imagenet'
        settings = pretrained_settings['xception'][pretrained]
        encoder.load_state_dict(model_zoo.load_url(settings['url']))
        encoder.input_space = settings['input_space']
        encoder.input_size = settings['input_size']
        encoder.input_range = settings['input_range']
        encoder.mean = settings['mean']
        encoder.std = settings['std']

    load=pre_gan
    if load:
        checkpoint = torch.load("C:/Users/Administrator/Desktop/VAE/tro/lby.pth")
        gan.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        #epoch = checkpoint['epoch'] + 1
    return 0
def train(i,images,labels):
    #net
    vector=encoder(images)
    output1, output, mn, sd = gan(vector)
    #loss
    img_loss = F.binary_cross_entropy(output, labels, size_average=False)
    latent_loss = -0.5 * torch.sum(1 + sd - mn.pow(2) - sd.exp())
    loss = img_loss + latent_loss
    #back
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(i, img_loss.cpu().detach().numpy(), latent_loss.cpu().detach().numpy(), loss.cpu().detach().numpy())
    #save
    if (i % 1000 <=3):
        print("save me!!!!!!!!!!!!!!!!")
        torch.save(state, "C:/Users/Administrator/Desktop/VAE/tro/lby.pth")
    return output