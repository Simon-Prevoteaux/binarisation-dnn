# binarisation-dnn
Aims to perform binarisation using DNN. We chose to use Crino, which is a ML lib using Theano.

Concerning the dataset, we first planed to use the DIBCO datas, but the images were to much different and we couldn't expect our model to generalize this type of data in the first. Therefore the dataset present on the repository is not used for now.
Instead, we used some newpapers images on which we performed a binarization using Sauvola's technique. At first, we expect to learn this model, even if the ground truth isn't perfect, it could give us the opportunity to see if our model is able to learn correctly.
