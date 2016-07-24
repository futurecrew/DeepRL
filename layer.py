"""The data layer used during training to train a Fast R-CNN network.

RoIDataLayer implements a Caffe Python layer.
"""

import caffe
import numpy as np

class DataLayer(caffe.Layer):
    """DQN data layer used for training."""
        
    def setup(self, bottom, top):
        """Setup the DataLayer."""

        # data blob: holds a batch of N images, each with 1 channels
        # The height and width (84 x 84) are dummy values
        top[0].reshape(1, 1, 84, 84)

        # labels blob: R categorical labels in [0, ..., K] for K foreground
        # classes plus background
        top[1].reshape(1)

            

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        """
        blobs = self._get_next_minibatch()

        for blob_name, blob in blobs.iteritems():
            top_ind = self._name_to_top_map[blob_name]
            # Reshape net's input blobs
            top[top_ind].reshape(*(blob.shape))
            # Copy data into net's input blobs
            top[top_ind].data[...] = blob.astype(np.float32, copy=False)
        """

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
