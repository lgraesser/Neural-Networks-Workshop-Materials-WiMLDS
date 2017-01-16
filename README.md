# Neural Networks Workshop

Materials for Neural Networks Workshop
* January 21 2017
* NYC Women in Machine Learning &amp; Data Science Meetup

## Installation Instructions

The instructions below will help you set up a virtual environment, get all of the scripts and data, and install all of the libraries required for this tutorial. There are two sets of instructions. If you have the Anaconda distribution of Python then follow the instructions "1. Using Anaconda". Otherwise follow the instructions "2. Using virtualenv."

What is a virtual environment and why use one?

*A virtual environment is a named, isolated, working copy of Python that maintains its own files, directories, and paths so that you can work with specific versions of libraries or Python itself without affecting other Python projects. Virtual environments make it easy to cleanly separate different projects and avoid problems with different dependencies and version requirements across components.* [Source](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/)

There are two ways to create virtual environments in Python.

1. Using the Anaconda distribution of Python.
2. Using virtualenv.
  
#### 1. Using Anaconda
If you have the anaconda distribution of Python (see [here](http://conda.pydata.org/docs/using/envs.html) for more info), then follow the instructions below. Note, there is an issue with matplotlib and python 3.6, so please specify the python version to be <=3.5 to make the matplotlib install work.

```shell
# In the terminal create a directory that you want to contain this repo and navigate to it
mkdir <directory_name>
# eg. mkdir NN_tutorial
cd <directory_name>
# Then, create a virtual environment
conda create -n <your_environment_name> python=<yourPythonVersion>
# for example conda create -n NN_tutorial python=3.5
# When conda asks you proceed ([y]/n)? type 'y'

# Switch into your new environment
    # Linux/OSX
    source activate <your_environment_name>
    # Windows
    activate <your_environment_name>

# Clone the repo, install the relevant libraries using setup.py and test the install worked
git clone https://github.com/lgraesser/Neural-Networks-Workshop-Materials-WiMLDS.git
cd Neural-Networks-Workshop-Materials-WiMLDS
python setup.py install
python test/test_install.py

# Now you are finished. 
# When you need to exit the environment (at the end of the tutorial for example), type.
    # Linux/OSX
    source deactivate
    # Windows
    deactivate
```

If you get an error message when running ```python test/test_install.py``` that ends with "Import Error: No module named tensorflow" this means that your version of Keras is using Tensorflow instead of Theano as the backend. To change the backend follow the instructions in the "Troubleshooting - Import Error: no module named tensorflow" section of this document.

#### 2. Using virtualenv
To use virtualenv (see [here](http://docs.python-guide.org/en/latest/dev/virtualenvs/) for more info) follow the instructions below.

```shell
# In the terminal, install virtualenv
pip install virtualenv
# Create a directory that you want to contain the repo and navigate to it
mkdir <directory_name>
# eg. mkdir NN_tutorial
cd <directory_name>
# Then, create a virtual environment
virtualenv <your_environment_name>
# Switch into your new environment
source <your_environment_name>/bin/activate

# Clone the repo, install the relevant libraries using setup.py and test the install worked
git clone https://github.com/lgraesser/Neural-Networks-Workshop-Materials-WiMLDS.git
cd Neural-Networks-Workshop-Materials-WiMLDS
python setup.py install
python test/test_install.py

# Now you are finished. 
# When you need to exit the environment (at the end of the tutorial for example), type.
deactivate
```

If you get an error message when running ```python test/test_install.py``` that ends with "Import Error: No module named tensorflow" this means that your version of Keras is using Tensorflow instead of Theano as the backend. To change the backend follow the instructions in the "Troubleshooting - Import Error: no module named tensorflow" section of this document.

### Installing Tensorflow

Keras is installed with Theano as the backend. This is all that is required for the tutorial. However, if you would like to install Tensorflow then [choose the correct binary to install from Tensorflow.](https://www.tensorflow.org/versions/r0.11/get_started/os_setup.html#pip-installation)

```shell
# for example, TF for Python3, MacOS, CPU only
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.11.0-py3-none-any.whl
# or, TF for Python2, MacOS, CPU only
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.11.0rc1-py2-none-any.whl
# or Linux CPU-only, Python3.5
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.11.0rc1-cp35-cp35m-linux_x86_64.whl

# Then install Tensorflow
    # Python 2
    $ sudo pip install --upgrade $TF_BINARY_URL
    # Python 3
    $ sudo pip3 install --upgrade $TF_BINARY_UR
```
Then you may need to change Keras' backend to Tensorflow. Check the current backend by typing into the terminal.

```
python -c "from keras import backend"
```
Either "Using Theano backend" or "Using tensorflow backend" will be printed to the screen. If the backend is "theano", then follow the instructions below in "Troubleshooting - Import Error: No module named tensorflow" to change it to tensorflow or see [Keras' documentation](https://keras.io/backend/).

### Troubleshooting

#### Import Error: No module named tensorflow

If you get an error message when running ```python test/test_install.py``` that ends with "Import Error: No module named tensorflow" this means that your version of Keras is using Tensorflow instead of Theano as the backend.

To change the backend to Theano you need to change the settings in the Keras config file.

First check if you have one by navigating to the directory listed below then typing ```ls``` to list the files contained in that directory. The commands are below.

```shell
cd ~/.keras/
ls
```

If there is a file named keras.json then open it by typing  ```open keras.json```  The default configuration looks like this.

```
{
    "image_dim_ordering": "tf",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "tensorflow"
}
```

Change the value of "backend" to "theano" and "image_dim_ordering" to "th". Save and close the file.

    
Alternatively, if there is no keras.json file then create one and open it by typing

```
touch keras.json
open keras.json
```

Then copy the information below into the file and save it.

```
{
    "image_dim_ordering": "th",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "theano"
}
```

Navigate back to the github directory (type ```cd ~``` in the terminal to navigate to your home directory, ```cd .. ``` to navigate to the directory that contains the directory you are currently in, and  ```ls``` at any point to list the files in the directory you are in). Then check that the installation tests works by running ```python test/test_install.py```

The first line printed to the screen when you run it should be "Using Theano backend"

#### Keras can be imported in Python but not IPython

If Keras is working in Python but not IPython, this is because the sys.paths of the two are different. Lucy Park has the answer. Follow [her tutorial](https://www.lucypark.kr/blog/2013/02/10/when-python-imports-and-ipython-does-not/) (only a few steps) and this should fix it

#### Still having issues?

Try installing with no virtual environments by following the instructions below

```shell
# In the terminal create a directory that you want to contain the repo and navigate to it
mkdir <directory_name>
# eg. mkdir NN_tutorial
cd <directory_name>

# Clone the repo, install the relevant libraries using setup.py and test the install worked
git clone https://github.com/lgraesser/Neural-Networks-Workshop-Materials-WiMLDS.git
cd Neural-Networks-Workshop-Materials-WiMLDS
python setup.py install
python test/test_install.py
```

## How to use this repo

There are three scripts which acommpany this tutorial. They can be run through either the jupyter notebook or the command line

* Part1
    * KerasIntro_Example1
    * KerasIntro_OtherExamples
* Part2
    * IntrotoNeuralNets2_CIFAR_2classes

```shell
# In the terminal, to make sure you have the most up to date version of the repo , type
git pull origin master

#To load an ipython notebook, type
jupyter notebook

# If you get an error, you may need to install ipython first. Try 
pip install ipython
# Then
jupyter notebook

# Otherwise to run a .py script from command line 
# navigate to the directory it is in and type python <filename>, for example
cd Part1
python KerasIntro_Example1.py
```

## Going further

* Read Michael Nielsen's excellent book [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)
* Watch over 20 hours of videos of deep learning tutorials thanks to the fantastic [Bay Area Deep Learning School](https://www.youtube.com/watch?v=eyovmAtoUx0). Day 2 is [here](https://www.youtube.com/watch?v=9dXiAecyJrY) and the schedule is [here](http://www.bayareadlschool.org/)
* Implement your own neural network. [See my implementation in Python](https://github.com/lgraesser/NeuralNetwork) for an example.
* Check out [my blog](https://learningmachinelearning.org/) for links to more resources on neural networks. I update it on a regular basis.
