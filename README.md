# CS578-Project

## To-do list

1. Tune hyperparameters (every algorithm)
2. Experiments: Iteration times vs. accuracy, ROC, etc (google sheet)
    - eg., SVM_C1.csv, SVM_C10.csv (",")
    - (Optional) Train_size: 50: 50: 450
    - Iteration_times: 50: 25: 500
3. Different features (raw pixel, facial landmark, raw + facial)
4. Different algorithms, CNN, decision tree
5. (Optional) PCA for dimension reduction (feature selection), AIC, BIC
6. T-test for K-fold and bootstrapping (different algorithms, using best parameters)


## Link of dataset
https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data

## Requirement
> The preliminary project report is due on Thursday November 5, 11.59pm EST
> (No extension days allowed, submission will open on November 2.)
> 
> On Brightspace, please submit a ZIP file with:
> 1) The report (in PDF or Word format) describing your progress so far.
> 2) Source code files.
> 
> I expect around 30-40% of what will be in your final report.
> 
> Therefore, at this point source code should include ALL the data preprocessing (e.g., reading the original format of the data, computing features, putting the data in table format, etc.), running SOME of the algorithms in your plan, and PART of the implementation of cross-validation. (Code for generating figures and tables of the experimental results such as charts, ROC curves, etc. are not required yet.)

> For the project, you will write a half-page project plan (around 1-2 weeks before the midterm), a 2-4 page preliminary results report (around 1-2 weeks after the midterm) and a 4-8 page final results report (around 1-2 weeks before the final exam). The project should include:
> 
> - a definition of the problem, possibly relevant to your interests.
> - a description of the dataset (or datasets) to be used. Datasets should be already publicly available (you should provide a URL), since there is not enough time for you to collect data. Possible datasets include: ADHD 200 (Whole Brain Data), Brain & Nouns, Connectomics, Higgs Boson, Labeled Faces in the Wild, Loan Default Prediction, Movielens, T-Drive, Yahoo Bidding (A1), Yahoo Ranking (C14).
> - a description of the experimental setup, e.g., cross-validation, parameter tuning, etc.
> - experimental results, showing not only when the algorithm succeeds but also when the algorithm fails. This might include: plots of number of samples versus accuracy (you can use different subsets of the same dataset), regularization parameter versus accuracy, ROC curves, plots of different datasets, etc.
> - you are allowed to either implement learning algorithms from scratch or use third-party code (e.g. liblinear). But ANY other thing such as cross-validation, parameter tuning, computing the values for the ROC curve, etc. should be written by yourself.
> - you can use either MATLAB, C++, Java or Python.
> - do not spend too much time on things such as "understanding the data", "memory problems because your data is too big", etc. Only if you are already familiar with computer vision, brain data, natural language processing, big data, parallelism, etc. then you can make use of those things, but this will not imply that you will get a higher grade just based on that fact. In general, I would recommend to use easy-to-understand datasets, and smaller subsets of the data, for instance.
> 
> Neither I nor the TAs will provide any help regarding programming-related issues.

## Labels
0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral.

## Dataset Description
* Source  
FER-2013 data was created by Pierre Luc Carrier and Aaron Courville, by using Google image search API to search for images of faces that match a set of 184 emotion-related keywords like "blissful", "enraged", etc.
Together with keywords like gender, age and ethnicity, about 600 strings were used to query facial images, and the first 1000 images return for each query were kept for the next stage of processing.

* Processing  
The collected images were approved by human labelers who removed incorrectly labeled images, cropped to only faces by bounding box utility of OpenCV face recognition, and resized to 48 x 48 pixels greyscale images. 
Then a subset of the images were chosen by Mehdi Mirza and Ian Goodfellow, and the labels(categories) of the chosen images were also mapped from the fine-grained emotion keywords.

* Structure  
The resulting FER-2013 dataset contains 35887 images with 7 categories in total. Specifically, there are 4953 “Anger” images, 547 “Disgust” images, 5121 “Fear” images, 8989 “Happiness” images, 6077 “Sadness” images, 4002 “Surprise” images, and 6198 “Neutral” images, with label ids ranging from 0 to 6.
![Distribution](https://github.com/Eroica-cpp/CS578-Project/blob/master/DatasetDistribution.png)

* Validation  
It was proven by Ian Goodfellow that the potential label errors in FER-2013 dataset do not make the classification problem significantly harder due to the experimental result that human accuracy on a small-scale dataset with 1500 images, 7 expression categories and no label error is 63-72%, which is very close to the human accuracy for FER-2013, which is 60-70%.  

## Tips
Use python pickle to save models.

## Results

## Reference
The problem of facial expression recognition has been studied extensively. 
Followings are some useful references:
- [Real-time Emotion Recognition From Facial Expressions](http://cs229.stanford.edu/proj2017/final-reports/5243420.pdf)
- https://github.com/amineHorseman/facial-expression-recognition-using-cnn
- https://github.com/amineHorseman/facial-expression-recognition-svm
- [Multiclass precision](https://miopas.github.io/2019/04/17/multiple-classification-metrics/)
- [Facial Expression Recognition using Convolutional Neural Networks: State of the Art](https://arxiv.org/pdf/1612.02903.pdf)
