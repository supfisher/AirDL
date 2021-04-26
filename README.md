# AirDL
This project aims at building a generic framework for distributed deep learning over the air. 
We build this simulator AirDL, which is compatible with Pytorch and Tensorflow. It can automatically trace the systematic QoS variables while training a deep learning model in the distributed scenario.


In practice, we will make an effort to realize the following requirements:
  1): Build a system model on the mobile edge network and wireless communication in the context of the current development of 5G and 6G.
  2): Realize the training of federated learning methods and algorithms.
  3): Evaluate our system by verifying the state-of-art of some typical applications.
  4): A handbook-like paper organizes and concludes all the aforementioned functions.
  

We are currently building the simulator with the NS-3. In the future, we will update it and isolate it from outside resources. 
1): In order to use it, you need to download the NS-3 at first https://www.nsnam.org/docs/tutorial/html/getting-started.html#downloading-ns-3-using-git. 
2): Move the files in Air into the contrib or src directory in NS-3. 
3): Move the files in demo into the scratch directory in NS-3.

