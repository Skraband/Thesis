import numpy as np
import torch

print(torch.cuda.is_available())

x = np.random.random((5,5))

y = np.random.random((10,10))

np.savetxt("test_data_folder/test_1/x",x)
np.savetxt("test_data_folder/test_2/y",y)