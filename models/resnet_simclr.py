import torch.nn as nn
import torchvision.models as models
import torch
from torch import Tensor

from exceptions.exceptions import InvalidBackboneError

class GumbelSigmoid(nn.Module):
    r"""Applies the element-wise function:

    .. math::
        \text{Sigmoid}(x) = \sigma(x) = \frac{1}{1 + \exp(-x)}


    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/Sigmoid.png

    Examples::

        >>> m = nn.Sigmoid()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    
    
    def __init__(self, temp : float = 0.4, eps:float =1e-10) -> None:
        super(GumbelSigmoid, self).__init__()
        self.temp = temp
        self.eps = eps
        
    def forward(self, input: Tensor) -> Tensor:
        # 
        if self.training:
            # input ma shape (batch,ilosc ilosc_wezlow_wew) -  (ilosc wezlow wew)
            # print(input.shape)
            # print(input.shape[1])
            uniform1 = torch.rand(input.shape[1]).cuda()
            uniform2 = torch.rand(input.shape[1]).cuda()
            
            noise = -torch.log(torch.log(uniform2 + self.eps)/torch.log(uniform1 + self.eps) +self.eps)
            
            #draw a sample from the Gumbel-Sigmoid distribution
            return torch.sigmoid((input + noise) / self.temp)
        else:
            return torch.sigmoid(input)
            

class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim, args=None):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, num_classes=out_dim),
                            "resnet34": models.resnet34(pretrained=False, num_classes=out_dim),
                            "resnet50": models.resnet50(pretrained=False, num_classes=out_dim)}
        # print(self._get_basemodel('resnet50'))
        self.f = []
        for name, module in models.resnet50().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # # projection head
        # self.g = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
        #                        nn.ReLU(inplace=True), nn.Linear(512, args.out_dim, bias=True))

        
        

        self.load_state_dict(torch.load('./pre-trained_models/new_cifar/128_0.5_200_128_1000_model.pth', map_location='cpu'), strict=False)


        sigmoidType = GumbelSigmoid(temp=args.temp) if args.gumbel else nn.Sigmoid() 
        self.tree_model = nn.Sequential(nn.Linear(2048, ((2**(args.level_number+1))-1) - 2**args.level_number), sigmoidType)
        self.f.to(args.device)
        self.tree_model = self.tree_model.to(args.device)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model

    def forward(self, x):
        x = self.f(x)
        x = torch.flatten(x, start_dim=1)
        return self.tree_model(x)
