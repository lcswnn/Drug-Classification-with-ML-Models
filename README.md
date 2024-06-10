# Drug-Classification-with-ML-Models
### Description
This project utilizes many different types of Classification Models, including K-Nearest Neighbors, Random Forest Classifier, Gaussian Naive Bayes, Decision Tree Classifier, MLP Classifier, and SVC Classifier. These were used to predict the type of drug per patient in the given data set. I first split the data into training and testing partitions, and fit them in each model. I then scored the models, printed those results out, and then graphed each performance against each other. We find that the Random Forset Classifier and the Decision Tree Classifier are both capable of having perfect performance most of the time, and a range of 0.96 - 1.0 each time you run it. 

**Libraries Used:** Pandas, Numpy, SKLearn (KNearestNeighbor, RandomForest, GaussianNB, DecisionTree, MLP, SVC, train_test_split), and MatPlotLib

Data used was found on Kaggle and formatted/one hot encoded in the jupyter notebook.
link: https://www.kaggle.com/datasets/prathamtripathi/drug-classification

### Findings
**Feature Findings:** When graphing the importance of the features using Random Forest Classification, we could see that the sex of the patient didn't matter as much as the other features. However, when we included it in our features list, the models had better performance rather than being overfit with features and having poorer performance, or just not making a difference. The features in the final product are the ones that gave the best performance per model. We also see that the Na_to_K feature in the list is by far the most important feature in the list, and could not be excluded from the tests.

**Other Findings:** We see that the Random Forest Classifier and the Decision Tree Classifier are the two superior models of this type of classification, as we see that both of the models are performing at 96%-100% accuracy, meaning both were tied for first. In second was K-Nearest Neighbors Classifier with 76% accuracy, and in third was the SVC model with a 74% accuracy, which aren't the worst scores, but definitely could be better. In fourth is the Gaussian Naive Bayes Classifier with 70% accuracy. And in last is the MLP Neural Network with 60% accuracy. This surprised me considering the model's performance on other data. I would guess there wasn't a lot of data for the neural network, there leading to a poorer performance due to lower training than required.

### Visualization of Model Performances
**Below is the graph of all models and their performances on a run**
