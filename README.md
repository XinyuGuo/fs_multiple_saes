Project: Feature Selection based on Multiple Sparse Autoencoders (SAEs)

In this project, a novel feature selection method based on multiple SAEs was designed to consturct 
a new SAE feature layer which can generate high-quality data representations, thus can ehance the 
performance in a classification task.

The method was examined in two classification tasks including a handwritten digit recogonition task,
and an object recogonition task. MNIST digit dataset (http://yann.lecun.com/exdb/mnist/), and CIFAR-10
image dataset (https://www.cs.toronto.edu/~kriz/cifar.html) was applied, respectively.

Three important foundings can be drawn from the resutls:

(1). The proposed feature selection method can improve the classification performance in both tasks. 
(2). The diversity of the selected feature pool is highly correlated with the classification model 
     performance.
(2). pinched-SAEs can produce more diverse features thus the classification accuracy can be improved.

More details and results of the method was described in the conferecne paper:

Guo, Xinyu, Ali A. Minai, and Long J. Lu. "Feature selection using multiple auto-encoders." 
In Neural Networks (IJCNN), 2017 International Joint Conference on, pp. 4602-4609. IEEE, 2017.
(http://ieeexplore.ieee.org/abstract/document/7966440/)

It's also a part of my Ph.D disertation.
