# Flower Image Classification - Using PyTorch
This project performs flower image classification. It was my submission for Facebook's PyTorch Challenge (warmly recommended!)

### Prerequisites & Installation

1. Note: a GPU is required to run this properly. If your computer does not have a GPU, you can either use a remote GPU machine (such as Google Colab) or run on a small number of images, though the latter yields low-quality results.

2. Install [Anaconda3](https://www.anaconda.com/distribution/) if you don't already have it: 

3. Download the [large flower dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/) and unzip anywhere on your computer, e.g. "/work/flower_data". If you cannot download, you can use the small dataset embedded in this project ("resources/flower_data_small"), though it would yield low-quality results.

4. Download this project directory anywhere on your local machine, e.g. "/work/flower_classification". Edit the code inside main_flower_classifier.py so that filenames point to the right directories on your local machine, e.g. the above mentioned image files "/work/flower_data"

5. Run the Anaconda shell to create and activate an environment (this example uses "my_env" but you can use any other name). The install the required packages:
```
conda create -n my_env python=3.7
activate my_env
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
pip install numpy
pip install torch
pip install torchvision
pip install matplotlib
```

## Running
From the same Anaconda shell (continuing with the activated "my_env") , change directory to the project's code and run it. For example:
```
cd /work/flower_classification/code
python main_flower_classifier.py
```
## Acknowledgments
Two great courses:
* Facebook's Pytorch Challenge Course
* [Coursera's Machine Learning Course](https://www.coursera.org/learn/machine-learning/home/welcome) by Prof. NG

