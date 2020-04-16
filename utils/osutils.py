from __future__ import absolute_import
import os
import errno
import torch
import pdb
from scipy.stats import truncnorm


def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
def truncated_z_sample(batch_size, z_dim, truncation=0.5, seed=None):
  state = None if seed is None else np.random.RandomState(seed)
  values = truncnorm.rvs(-2, 2, size=(batch_size, z_dim), random_state=state)
  return truncation * values

def get_vector(inputs,model,layer_num):
    
    # 3. Create a vector of zeros that will hold our feature vector
    #    The 'avgpool' layer has an output size of 512
    #my_embedding = torch.zeros(32,512)
    #pdb.set_trace()
    #pdb.set_trace()
    if layer_num == 0:
        tensor_size = [128,64,32,32]
        layer = model._modules.get('layer1')[1]._modules.get('bn2')
    elif layer_num ==1:
        tensor_size = [128,128,16,16]
        layer = model._modules.get('layer2')[1]._modules.get('bn2')
    elif layer_num ==2:
        tensor_size = [128,256,8,8]
        layer = model._modules.get('layer3')[1]._modules.get('bn2')
    elif layer_num ==3:
        tensor_size = [128,512,4,4]
        layer = model._modules.get('layer4')[1]._modules.get('bn2')
    elif layer_num ==4:
        pass

    my_embedding = torch.zeros(tensor_size)
    #my_embedding = torch.zeros(32,128,28,28)
    #my_embedding = torch.zeros(32,256,14,14)
    # 4. Define a function that will copy the output of a layer
    def copy_data(m, i, o):
        my_embedding.copy_(o.data)
    # 5. Attach that function to our selected layer
    
    
    
    h = layer.register_forward_hook(copy_data)
    # 6. Run the model on our transformed image
    model(inputs)
    # 7. Detach our copy function from the layer
    h.remove()
    # 8. Return the feature vector
    return my_embedding