# CS578-Project

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

