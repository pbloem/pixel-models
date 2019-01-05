# pixel-models
Pytorch implementations of the PixelCNN (va Oord et al. 2016) and PixelVAE (Gulrajani et al. 2016) models.

**STATUS**: This project is not finished. Most models are implemented, but they have not been fully tested, to show that they reproduce the performane of the published versions. If you are doing something important, please don't use this code as is.

## Usage notes

Make sure that the value of the "current pixel" (at the center of the convolution), is not transmitted up the network:
 * Disable self-connections in the first layer
 * For the gated layer, disable the residual connection in the first layer.
 
## Tasks (i.e. datasets)

Most datasets are downloaded automatically. For the imagenet64 dataset, download the following torrent: 
[Academic torrents](http://academictorrents.com/details/96816a530ee002254d29bf7a61c0c158d3dedc3b)
 [magnet link](https://goo.gl/nr7NFi) 
Place the train and test data in some directory (say ```~/data/imagenet/```), containing subdirectories ```train``` and ```valid```. These should each 
contain an additional subdirectory (with any name), which contains the images.
 
Then call the training script as follows:
```python train-cnn.py -t imagenet64 -D ~/data/imagenet/```

## Sources

Some of the code in this repository was adapted from:    
  
* https://github.com/j-min/PixelCNN

## Implementation notes

I noted the following things during implementation. These may be useful if you're trying to port or adapt this code.

* If the model has access to pixel x in its input, for its prediction of pixel x, validation loss will be very low, 
but the samples will converge to black images. This happened, for instance, when I accidentally had a skip connection
in the first layer. If your samples go black, check that the model can't accidentally see the pixel it's predicting.

* The model allows colors to be conditioned sequentially. For instance, when sampling the green value for a pixel, 
we are allowed to see the red value. This seems like a minor improvement, but it's crucial for good performance. Without
it, there is little to no coordination between the colors and the picture looks like random noise.

