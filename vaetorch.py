import torch
import trainer
import matplotlib.pyplot as plt

torch.manual_seed(1)    # reproducible
EPOCH = 1           # 训练整批数据多少次, 为了节约时间, 我们只训练一次
batch_size =2
trainer.pretrain(pre_encoder=True,pre_gan=True)
i=1
while (i<=2975):
    images,labels,i=trainer.data(i)
    output=trainer.train(i,images,labels)
    imgs=output.reshape(-1,128,256,4)
    plt.imshow(imgs[0].cpu().detach().numpy())
    plt.savefig("vaet/out-%d.png" %((i-1)%1000+1))
    plt.close()
    plt.imshow(imgs[1].cpu().detach().numpy())
    plt.savefig("vaet/out-%d.png" % (i % 1000 + 1))
    plt.close()