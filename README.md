# ktree

This GitHub repository provides the code for the thesis project: 

### "Analysis of the computational power of single neurons - Resolution of classification problems"

by [L. Petrella](mailto:lorenzo.petrella@mail.polimi.it), [M. Protopapa](mailto:margherita.protopapa@mail.polimi.it) and [P. Velcich](mailto:pietro.velcich@mail.polimi.it) (BSc in Biomedical Engineering @ Politecnico di Milano) under the supervision of [Prof. D. Linaro](mailto:daniele.linaro@polimi.it).

#### Abstract:
Since the dawn of Machine Learning, Deep Learning algorithms have drawn inspiration from the functioning and morphology of biological neural networks. However, state-of-the-art models today treat each neuron as a single point unit, representing the soma, outputting a specific function of its inputs, thus entirely overlooking the potential computational role of the often complex and deep dendritic tree. The project presented here aims at reproducing and examining in greater depth the study carried out by Jones and Kording in 2021 with respect to the computational power of single neurons, modeled as a soma to which a dendritic tree consisting of one or more subtrees, built as binary trees, is connected as input.
For that purpose, a suitable supervised Deep Learning model has been implemented with TensorFlow, drawing inspiration from the architecture of single neurons, for the binary classification of images; such model has then been employed to discriminate between images belonging to several datasets, such as MNIST, which are largely used in the field of Machine Learning for the assessment of the computational power of classifiers.
Having verified the correspondence between the results obtained with the implemented model and those reported in the article by Jones and Kording, the study proceeds with the assessment of the impact on overall performance of variations in morphology, distribution of inputs and constraints on the parameters, partly relying as a starting point on the prospective investigations outlined by Jones and Kording. Additionally, the possibility of building complex dendritic trees by assembling simpler pretrained subtrees is considered, with the goal of reducing the duration of the training phase without impacting significantly the performance of the model.
The main results obtained throughout the project show that repeating inputs through different image unrolling techniques guarantees, depending on what the pictures represent, a significant increase in the accuracy of the classification. Furthermore, unlike Jones and Kording, the asymmetry of the dendritic tree, which is more realistic from a biological standpoint, has been proven not to deteriorate the performance of the model. It has also been observed that building a model by assembling simpler dendritic trees, despite greatly reducing the training time thanks to careful expedients, determines a negligible or otherwise contained decrease in the goodness of the classifier. Lastly, although generally unjustified from a biological point of view, the addition of a biasing contribution at each fork of the dendritic tree, as well as the possibility for trainable weights to assume negative values, have been shown to cause a significant increase in the performance of the model.


#### Credits:

This project is based on the previous study conducted by I.S. Jones and K.P. Kording, "Can Single Neurons Solve MNIST? The Computational Power of Biological Dendritic Trees". Part of their code has been used and/or adapted for the purposes of this thesis.
