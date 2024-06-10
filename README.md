# Drug-Classification-with-ML-Models
### Description
This project utilizes many different types of Classification Models, including K-Nearest Neighbors, Random Forest Classifier, Gaussian Naive Bayes, Decision Tree Classifier, MLP Classifier, and SVC Classifier. These were used to predict the type of drug per patient in the given data set. I first split the data into training and testing partitions, and fit them in each model. I then scored the models, printed those results out, and then graphed each performance against each other. We find that the Random Forset Classifier and the Decision Tree Classifier are both capable of having perfect performance most of the time, and a range of 0.96 - 1.0 each time you run it. 

**Libraries Used:** Pandas, Numpy, SKLearn (KNearestNeighbor, RandomForest, GaussianNB, DecisionTree, MLP, SVC, train_test_split), and MatPlotLib

Data used was found on Kaggle and formatted/one hot encoded in the jupyter notebook.
link: https://www.kaggle.com/datasets/prathamtripathi/drug-classification

### Findings
**Feature Findings:** When graphing the importance of the features using Random Forest Classification, we could see that the sex of the patient didn't matter as much as the other features. However, when we included it in our features list, the models had better performance rather than being overfit with features and having poorer performance, or just not making a difference. The features in the final product are the ones that gave the best performance per model. We also see that the Na_to_K feature in the list is by far the most important feature in the list, and could not be excluded from the tests.

**Other Findings:** Model Rankings:
1. Random Forest Classifier/Decision Tree Classifier
2. MLP Neural Network
3. K-Nearest Neighbors Classifier
4. SVC
5. Gaussian Naive Bayes Classifier

(*Note: Due to the Random Forest Classifier being made up of many Decision Trees, it makes sense for the two models to have the same performance. However, they're not identical. If we ran further tests, my prediction would be that the Random Foest Classifier would have a higher average than the Decision Tree Classifier due to being made up of multiple Decision Trees, rather than just having one. This would mainly prevent overfitting in the model.*)

### Visualization of Model Performances
**Below is the graph of all models and their performances on a run**
![output](https://github.com/lcswnn/Drug-Classification-with-ML-Models/assets/118494460/422440fa-8dc5-4424-a288-6d2b7b7d955f)
