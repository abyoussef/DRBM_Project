This is the structure of the code that implements the Multimodal Deep Boltzmann Machine :

* **config.py** a python configuration file that store the directory of the data and the directory to store the learned model.

* **rbm.py** a python class that implements an RBM. It supports the binary and gaussion type of RBMs. 
 I added to it another type, in order to include the Replicated Softmax Model of RBMs.

* **dbm.py** a python class that implements a Deep Boltzmann Machine. It is implemented as stacked RBMs. 

* **run_flickr_text_dbm.ipynb** is the python script that trains the text specific Deep Boltzmann Machine.  
  It outputs **text_dbm_layer_i_W.npy** and **text_dbm_layer_i_b.npy** for layer i. 

* **run flickr image dbm.ipynb** is the python script that trains the image specific Deep Boltzmann Machine. 
 It outputs **image_dbm_layer_i_W.npy** and **image_dbm_layer_i_b.npy** for layer i.

* **run_flickr_multimodal_dbm.ipynb** is the python script that implements the multimodal Deep Boltzmann Machine using the weights learned from the two precedent scripts.
