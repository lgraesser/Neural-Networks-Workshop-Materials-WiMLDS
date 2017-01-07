# Neural Networks Workshop Materials WiMLDS

Materials for Neural Networks Workshop
* January 21 2017
* NYC Women in Machine Learning &amp; Data Science Meetup

## Installation

### Basic

* Create a directory to contain this repository
* Navigate to that directory and run the commands below. This will install all the relevant libraries needed for the tutorial and to run the scripts contained in this repository. 
	* The cleanest way to install all of the libraries is to create a virtual environment and install them within there. Details for how to do this are in the troubleshooting section. However this is not required.

```shell
git clone https://github.com/lgraesser/Neural-Networks-Workshop-Materials-WiMLDS.git
cd Neural-Networks-Workshop-Materials-WiMLDS
python setup.py install
python test/test_install.py
```

### Tensorflow

Keras is installed with Theano as the backend. If you would like to install Tensorflow then [choose the correct binary to install from TF.](https://www.tensorflow.org/versions/r0.11/get_started/os_setup.html#pip-installation)

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
Then you will need to [change Keras' backend](https://keras.io/backend/) to Tensorflow

### Troubleshooting

If you run into installation or import errors, first try creating a virtual environment, cloning the repo and running setup.py from inside this environment. 

If you have the anaconda distribution (see [here](http://conda.pydata.org/docs/using/envs.html) for more info).

```shell
conda create -n <your_environment_name> python=<yourPythonVersion>
# for example conda create -n NN_tutorial python=3
# When conda asks you proceed ([y]/n)? type 'y'
# Switch into your new environment
# Linux/OSX
source activate <your_environment_name>
# Windows
activate <your_environment_name>
# Then follow previous instructions
git clone https://github.com/lgraesser/Neural-Networks-Workshop-Materials-WiMLDS.git
cd Neural-Networks-Workshop-Materials-WiMLDS
python setup.py install
python test/test_install.py
# To exit the environment
# Linux/OSX
source deactivate
# Windows
deactivate
```

Otherwise you can use virtual environment (see [here](http://docs.python-guide.org/en/latest/dev/virtualenvs/) for more info).

```shell
# Install if needed
pip install virtualenv
# Create a virtual environment
cd <folder_name>
virtualenv <your_environment_name>
# Switch into your new environment
source <your_environment_name>/bin/activate
# Then follow previous instructions
git clone https://github.com/lgraesser/Neural-Networks-Workshop-Materials-WiMLDS.git
cd Neural-Networks-Workshop-Materials-WiMLDS
python setup.py install
python test/test_install.py
# To exit the environment
deactivate
```

If Keras is working in Python but not IPython, this is because the sys.paths of the two are different. Lucy Park has the answer. Follow [her tutorial](https://www.lucypark.kr/blog/2013/02/10/when-python-imports-and-ipython-does-not/) (only a few steps) and this should fix it

## Usage

There are two scripts which can be run through either the ipython notebook or the command line
* Part1/KerasIntro_Example1.py/ipynb
* Part1/KerasIntro_OtherExamples.py/ipynb

The last script is only available with the ipython notebook
* Part2/IntrotoNeuralNets2_CIFAR_2classes.ipynb

Scripts can be run using an ipython notebook or the command line
```shell
# To load ipython notebook
ipython notebook
# Otherwise to run from command line, for example
python Part1/KerasIntro_Example1.py
```

## Going further

* Read Michael Nielsen's excellent book [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)
* Implement your own neural network. [See my implementation in Python](https://github.com/lgraesser/NeuralNetwork) for an example.
* Check out [my blog](https://learningmachinelearning.org/) for links to more resources on neural networks. I update it on a regular basis.
