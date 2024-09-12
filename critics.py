''' LIBRARIES '''
import torch
from torch import nn
from torchvision.models import densenet, resnet
import torch.nn.functional as F




class Mish(nn.Module):
  def forward(self, x):
    return x * torch.tanh(F.softplus(x))


''' Mish-based Critics, in a Basic, Residual, and Dense implementation '''
# BASIC: all returned as one sequential layer
class BasicMishCritic(nn.Module):
  def _conv2d(self, in_channels, out_channels): # identical to SteganoGAN
      return nn.Conv2d(
          in_channels=in_channels,
          out_channels=out_channels,
          kernel_size=3
      )


  ''' Build models (modified):
  Instead of directly creating the
  '''
  def _build_models(self):
    self.c1 = nn.Sequential( # 1
      self._conv2d(3, self.hidden_size),
      Mish(),
      nn.BatchNorm2d(self.hidden_size)
    )

    self.c2 = nn.Sequential( # 2
      self._conv2d(self.hidden_size, self.hidden_size),
      Mish(),
      nn.BatchNorm2d(self.hidden_size)
    )

    self.c3 = nn.Sequential( # 3
      self._conv2d(self.hidden_size, self.hidden_size),
      Mish(),
      nn.BatchNorm2d(self.hidden_size)
    )

    self.c4 = nn.Sequential( # 4
      self._conv2d(self.hidden_size, 1)
    )

    return self.c1, self.c2, self.c3, self.c4 # return the values as a tuple

  def __init__(self, hidden_size):
    super().__init__()
    self.hidden_size = hidden_size
    self._models = self._build_models()

  def forward(self, img): # new feed forward loop manually moves from one layer to the next
    x = self._models[0](img)
    x1 = self._models[1](x)
    x2 = self._models[2](x1)
    x3 = self._models[3](x2)
    final_x = torch.mean(x3.view(x3.size(0), -1), dim=1)

    return final_x

# DENSE
class DenseMishCritic(nn.Module):
  def __init__(self, weights=densenet.DenseNet121_Weights.IMAGENET1K_V1):
    super(DenseMishCritic, self).__init__() # initialize using inheritance
    self._models = densenet.densenet121(weights=weights)
    self._models.train()

  def forward(self, x):
    features = self._models.features(x)
    out = Mish()(features)
    out = F.avg_pool2d(out, kernel_size=7).view(features.size(0),-1)
    out = torch.mean(out.view(out.size(0),-1),dim=1)
    return out

# RESIDUAL
class ResidualMishCritic(nn.Module):
  def __init__(self, num_classes=2):
    super(ResidualMishCritic, self).__init__() # initialize using inheritance
    self._models = resnet.ResNet(resnet.BasicBlock, [2,2,2,2], num_classes=num_classes)
    self.replace_relu_with_mish()
    self._models.train()

  def forward(self, x):
    x = self._models.conv1(x)
    x = self._models.bn1(x)
    x = Mish()(x) # replace ReLU with Mish
    x = self._models.maxpool(x)

    x = self._models.layer1(x)
    x = self._models.layer2(x)
    x = self._models.layer3(x)
    x = self._models.layer4(x)

    x = self._models.avgpool(x)
    x = torch.mean(x.view(x.size(0), -1), dim=1)
    return x

  # Helper method
  def replace_relu_with_mish(self):
    for name, module in self._models.named_children():
      if isinstance(module, nn.ReLU):
        setattr(self._models, name, Mish())
      elif isinstance(module, nn.Sequential):
        for child_name, child_module in module.named_children():
          if isinstance(child_module, nn.ReLU):
            setattr(module, child_name, Mish())