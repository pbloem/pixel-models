# pixel-models
Pytorch implementations of the PixelCNN (va Oord et al. 2016) and 
PixelVAE (Gulrajani et al. 2016) models.

## Usage notes

Make sure that the value of the "current pixel" (at the center of the convolution), is not transmitted up the network:
 * Disable self-connections in the first layer
 * For the gated layer, disable the residual connection in the first layer.

## Sources

Some of the code in this repository was adapted from:    
  
* https://github.com/j-min/PixelCNN

## Implementation notes

I noted the following things during implementation. These may be useful if you're trying to port or adapt this code.

* If the model has access to pixel x in its input, for its prediction of pixel x, validation loss will be very low, 
but the samples will converge to black images. This happened, for instance, when I accidentally had a skip connection
in the first layer. If your samples go black, check that the model can accidentally see the pixel it's predicting.

* The model allows colors to be conditioned sequentially. For instance, when sampling the green value for a pixel, 
we are allowed to see the red value. This seems like a minor improvement, but it's crucial for good performance. Without
it, there is little to no coordination between the colors and the picture looks like random noise.

