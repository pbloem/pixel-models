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



## Noted during implementation

I noted the following things during implementation. These may be useful if you're trying to port or adapt this code.

* If you the model has access to pixel x in its input, for it's prediction of pixel x, validation loss will be very low, 
but the samples will converge to black images. This happened, for instance, when I accidentally had a skip connection
in the first layer. If your samples go black, check that the model can accidentally see the pixel it's predicting.