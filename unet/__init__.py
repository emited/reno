from .unet_model import UNet

class UnetPermuted(UNet):
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = super(UnetPermuted, self).forward(x)
        x = x.permute(0, 2, 3, 1)
        return x