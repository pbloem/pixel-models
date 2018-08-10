# pixel-models
Pytorch implementations of the PixelCNN (va Oord et al. 2016) and 
PixelVAE (Gulrajani et al. 2016) models.

## Notes

Make sure that the value of the "current pixel" (at the center of the convolution), is not transmitted up the network:
 * Disable self-connections in the first layer
 * For the gated layer, disable the residual connection in the first layer.

## Sources

Some of the code in this repository was adapted from:    
  
* https://github.com/j-min/PixelCNN



