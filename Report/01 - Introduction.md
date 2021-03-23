# Data Science Toolbox - Assessment 04

*This project is the 4th assessment for the Data Science Toolbox from the MSc Mathematics of Cybersecurity course at the University of Bristol (2020/21). There are 4 contributors to the project. While it is preferrable to be able to test models and draw solid conclusions, the most important parts of these assessments is to explore options available for good Data Science in the industry and learn new skills throughout the project.*

## Introduction

In this assessment our group was tasked to apply Deep Learning in a cyber-security context. Similar to the previous assessment, the actual brief this time was rather short and most of the decisions were left up to us as a group to decide. 

The first decision we had to make was what was going to be the main focus of our work. In order to help with this decision we decided to research into potential datasets we could use in our models. In doing so we re-discovered the [UNSW-NB15 Dataset](https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/), that some of us had seen before in previous assessments. This data features a large amount of features with a classified "Attack Category" at the end. Interestingly, the last time the four of us had worked together in a group was on Assessment 01, where we tackled a classification problem using various methods. We thought that it would be interesting to try and apply the deep learning power of Neural Networks to this problem to see how well they coped with classifying data like this.

First of all, we downloaded and compressed the four parts of the dataset to our GitHub repository as the files on the datasets website were uncompressed and took a long time to download; it is also easy to access from within our own repository and can be found in the [Data folder](https://github.com/Galeforse/DST-Assessment-04/tree/main/Data). 

We decided that we would attempt to use different forms of Neural Network implementations and see how they compared, as well as get a general impression for how well Neural Networks performed on a classification task such as this. This immediately limited some of the networks available to us as different forms of network are used more in different application such as image classification, which did not apply in our case. We therefore decided to experiment with just a few different implementations as well as some differences in parameters of similar networks in order to see how this affected the final result.

Due to the inherent complexity of Deep Learning (though often made easier throught he implementation of packages), there were two major hurdles that we would have to address in this project:

* Would our machines be up to the challenge of running a Neural Network? The main issues we were concerned about were memory usage, as well as the potential time it might take a model to run.
* Making sure the data was suitable for use in a Neural Network, making sure we followed guidlines on how to pre-process our data for correct implementation.

To address the first point, in this assessment we made use of the University of Bristol's high performance computing machine: [BlueCrystal](https://www.acrc.bris.ac.uk/acrc/phase4.htm) (Of which we used phase 4). This was a challenge for our group to overcome, but after a lot of experimentation we now feel that we have learnt a new skill which will help in future endeavours, both in the rest of our course and beyond. Use of the HPC allowed us to not only run scripts with better system specs but also to run overnight and while we were away from our PCs allowing long deep learning to take place.

Through what we have learned in lectures and additional research from documentation of packages that implement Neural Networks we developed a good data pre-processing pipeline that would help with getting better results. Much of the data processing took place individually so that we could export our code for use on the HPC; this was also in part due to using different coding languages in our group resulting in potential discrepancies in the form of data.

In each section of the report we will explain briefly explain the differences in each model that we have implemented along with our code. In the [Performance Analysis](https://github.com/Galeforse/DST-Assessment-04/blob/main/Report/06%20-%20Performance%20Analysis.ipynb) document we compare the results achieved from each model and will then draw conclusions about the usage of Neural Networks for this kind of classification problem.

The reader is asked to look at each item in the [Report](https://github.com/Galeforse/DST-Assessment-04/tree/main/Report) folder in the designated number order to follow a similar sequence of events to how we conducted our code. However some code was shared amongst the group at the same time, and therefore several sections will make reference to parts that feature later in the report.

## Contributors

Additional annotated and unannotated work can be find in our individual folders, but not all of this made it into the final report:

[Alex Caian](https://github.com/Galeforse/DST-Assessment-04/tree/main/Alex%20Caian)

[Gabriel Grant](https://github.com/Galeforse/DST-Assessment-04/tree/main/Gabriel%20Grant)
    
[Luke Hawley](https://github.com/Galeforse/DST-Assessment-04/tree/main/Luke%20Hawley)

[Matt Corrie](https://github.com/Galeforse/DST-Assessment-04/tree/main/Matt%20Corrie)
