
# Introducing Python Libraries
## Objectives
* Understand and be able to explain what a library is and why is it important
* Understand and explain the purpose of the key data science libraries in Python (Numpy, Pandas, Seaborn, Matplotlib, SciPy and Scikit-learn, tensorflow, keras and statsmodels)


##  Introduction

A library (or a module/package) is a piece of software that adds a specific functionality to programming platforms in addition to functions already available. For example, when running an analytics experiment, data scientists would bring in a number of modules to python environment for a multitude of purposes. For example, one module would be responsible for reading and processing external data in a convenient manner, another module would be responsible for plotting visualizations without defining pixel level details, and another one for running complex machine learning algorithms which would be highly non-trivial to code otherwise. 

In this lesson, we shall look at some of the key libraries used in python for purpose of advanced data analytics. You will come across these (and some more libraries) during the course. **Remember** we are only looking at data science specific libraries at this stage. Python offers many other good libraries for say game development, 3D modelling, web applications etc. 

So here's a first quick intro to analytics libraries to get you guys going. 

## Python Libraries for Data Science

Due to the fact that python is an open source programming environment, a number libraries have been developed for python to enhance its core functionality over the years. Python has gained a lot of traction in the recent years in the Data Science domain. 

Following image highlights professional libraries which are routinely used for data analysis. 

![](https://www.enthought.com/wp-content/uploads/Canopy-Packages-min-1.png)

Let's look at some of the key libraries for data scientists and engineers. 

## Scientific Computation

The key computing need for a data scientist is to be able to convert data in an easy to process format, or the ability to convert data from multiple formats into standard formats. Data, represented inside a computer may become too complex to be processed by lists and dictionaries and using python's built in methods, which may not be suitable for mission critical, high performance and precision tasks. Following libraries provide scientific computation abilities to python. Let's have a quick look at some of these. 

### Numpy 


In python, the most fundamental package used for scientific computation is **NumPy** (Numerical Python). It provides lots of useful functionality for mathematical operations on vectors and matrices in Python. Matrix computation, is considered the primary strength of NumPy. 

![](https://1.bp.blogspot.com/-CHMzy5L0Qcw/Wpy00BAw-dI/AAAAAAAAG9g/fBinxajEzcshsZPSemZIt37JlqOWdDWbQCLcBGAs/s1600/numpy.jpeg)


The library provides these mathematical operations using NumPy **array** data type, which enhances performance and speeds up the execution as compared to python default processing. It contains among other things:

* A powerful N-dimensional array object
* Sophisticated (broadcasting) functions
* Tools for integrating C/C++ and Fortran code
* Useful linear algebra, Fourier transform, and random number capabilities
* Besides its obvious scientific uses, NumPy can also be used as an efficient multi-dimensional container of generic data. Arbitrary data-types can be defined. This allows NumPy to seamlessly and speedily integrate with a wide variety of databases.

Numpy code is used a foundations to other more advanced libraries as we shall see below.

### SciPy

In the data science domain, Python’s SciPy Stack ( a collection of software specifically designed for scientific computing ) is used heavily for conducting scientific experiments. SciPy library is as integral part of this stack.

<img src="http://scipy.in/2015/static/img/scipy.png" width = 150>

SciPy is a library of software for engineering and science applications and contains functions for **linear algebra**, **optimization**, **integration**, and **statistics**. 

The functionality of SciPy library is built on NumPy, and its data structures make heavy use of NumPy. It provides efficient numerical computational routines and comes packaged with a number of specific submodules. Following are a few modules from this library which are very commonly applied to data science experiments.

* stats: statistical functions
* linalg: linear algebra routines
* fftpack: Discrete Fourier Transform algorithms
* signal: signal processing tools
* optimize: optimization algorithms including linear programming

### Statsmodels 
Statsmodels is a library for Python that enables its users to conduct data exploration via the use of various methods of estimation of statistical models and performing statistical assertions and analysis.

![](https://numfocus.org/wp-content/uploads/2017/11/statsmodels-logo-300.png)

Among many useful features are descriptive statistics. The library given insight information as diagnostics with linear regression models, generalized linear models, discrete choice models, robust linear models, time series analysis models with various estimators.

The library also provides extensive plotting functions that are designed specifically for the use in statistical analysis and tweaked for good performance with big data sets of statistical data.

### Pandas

Pandas is a Python package designed to do work with “relational” data and helps replicates the functionality of relation databases in a simple and intuitive way. Pandas is a great tool for data wrangling. It designed for quick and easy data cleansing, manipulation, aggregation, and visualization.
![](http://www.howcsharp.com/img/1/47/pandas-300x300.jpg)

There are two main data structures in the library: 

1. “Series” — one-dimensional
2. “Data Frames” - two-dimensional

These data types can be manipulated in a number of ways for analytical needs. Hereare a few ways in which pandas may come in handy:

* Easily delete and add columns from DataFrame
* Convert data structures to DataFrame objects
* Handle missing data and outliers
* Powerful grouping and aggregation functionality
* Offers visualization functionality under the hood to plot complex statistical visualizations on the go
* The data structures in pandas are highly compatible with most of other libraries. 





## Data Visualization

Data visualization and visual analytics of the data in addition to predictive analytics are routine tasks that data scientists come across. Traditionally, drawing visualizations would involve providing pixel level details and complex mathematical functions to govern the visual aspect. Luckily, Python has good library support for data visualization from plotting routine visualizations in matplotlib, to developing graphical dashboards in Plotly and Bokeh etc. In this course, we shall cover following graphical packages.

### MatplotLib


Matplotlib is another SciPy Stack package and a library that is tailored for the generation of simple and powerful visualizations with ease. It is a sophisticated package which is making Python (with some help of NumPy, SciPy, and Pandas) an industry standard analytics tool. 

![](https://matplotlib.org/_static/logo2.png)

Matplotlib is a flexible plotting library for creating interactive 2D and 3D plots that can also be saved as manuscript-quality figures. The API in many ways reflects that of MATLAB, easing transition of MATLAB users to Python. Many examples, along with the source code to re-create them, are available in the matplotlib gallery. With a bit of effort you can make just about any visualizations:
```
Line plots
Scatter plots
Bar charts and Histograms
Pie charts
Stem plots
Contour plots
Quiver plots
Spectrograms

```
There are also facilities for creating labels, grids, legends, and many other formatting entities with Matplotlib. Basically, everything is customizable.

The library, however,  is pretty low-level which means that you will need to write more code to for advanced visualizations and will generally need more effort.

### Seaborn 

Seaborn is complimentary to Matplotlib and it specifically targets statistical data visualizations, which maybe more time-consuming in Matplotlib. Seaborn extends the functionality od Matplotlib and that’s why it can address the two biggest issues of working with Matplotlib i.e. Quality of plots an Parameter defaults.


<img src="https://ksopyla.com/wp-content/uploads/2016/11/seaborn_examples.jpg" width=500>

>If matplotlib “tries to make easy things easy and hard things possible”, seaborn tries to make a well-defined set of hard things easy too.

As Seaborn compliments and extends Matplotlib, the learning is very fast. If you know Matplotlib, you’ll already have most of Seaborn down. your plots with seaborn will more pretty, need less time and reveal more information. 

## Machine Learning 
### Scikit-Learn 

Scikits provide Scientific "kits" on top of SciPy Stack. These are designed to add specific functionality to SciPy like image processing and machine learning facilitation. For machine learning, one of the most heavily used package is **scikit-learn**. The package is built on makes heavy use of its maths operations to model and test complex computational algorithms.

<img src="https://www.scipy-lectures.org/_images/scikit-learn-logo.png" width=300>

The scikit-learn offers a consistent interface to the common machine learning algorithms, making it simple to bring ML into production systems. The library combines quality code and good documentation, ease of use and high performance and has become industry standard for machine learning with Python. The image below highlights the key machine learning algorithms that come packaged with sklearn for problems in classification, regression, clustering and dimensionality reduction.  In this course, we shall mainly use this library for our experiments including most of the algorithms shown below. 


![](http://1.bp.blogspot.com/-ME24ePzpzIM/UQLWTwurfXI/AAAAAAAAANw/W3EETIroA80/s1600/drop_shadows_background.png)

## Deep Learning  ( Keras / TensorFlow )

In the regard of Deep Learning, one of the most prominent and convenient libraries for Python is Keras, which can function on top of TensorFlow. Let's go through some key details about both of these libraries.

### TensorFlow

Developed by team of ML experts at Google, TensorFlow is an open-source library of data flow graphs computations (also called graph computations), which are fine tuned for heavy duty Machine Learning. TensorFlow was designed to meet the performance requirements of Google environment for training Deep Neural Networks in order to analyze visual and textual data. However, TensorFlow isn’t strictly for scientific use in border’s of Google — it is general enough to use it in a variety of real-world application.
![](https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2018/01/tf.png)


The key feature of TensorFlow is their multi-layered nodes system that enables quick training of artificial neural networks on big data. This is the library that powers Google’s voice recognition and object recognition in real time. 



### Keras

Keras is an open-source library for building Neural Networks with a high-level of interface abstraction. Keras library is written in Python so python developers find it much easier to start coding for deep networks in Keras, than Tensorflow, which demands a good understanding of graph computation. Keras is much more minimalistic and straightforward with high-level of extensibility. Under the hood, It uses Theano ( another deep learning library) or TensorFlow.

![](https://img.itw01.com/images/2018/03/30/01/3831_z2Vv5Y_VHWRKEI.jpg!r1024x0.jpg)
The minimalistic approach in design aimed at fast and easy experimentation through the building of compact systems.

Keras is really eased to get started with and for quick prototyping. It is highly modular and extendable. Notwithstanding its ease, simplicity, and high-level orientation, Keras is still deep and powerful enough for serious modeling. In the deep learning section of our course, we shall introduce you to Keras to help you deep dive into deep neural networks without having to understanding a different development paradigm like tensor handling and computation.

## Summary 

This lesson provides a brief introduction to some of the python libraries that are commonly used in data science discipline. We shall be looking at these libraries at some point in the course, starting from Pandas and matplotlib for data wrangling and visualization. Other libraries will be introduced with t
