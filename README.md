# ImageClassifier

- original image size (1 x 10 x 10), it is symmetric, use only the right-top quarter 
- 0: success, 1: failure (based on the top 20 quantile value of bandgap)
- imbalanced data (80/20)

## Models: 
1. CNN with data augmentation (i.e., random flip and random rotation) and imbalanced data sampler(pytorch, TLClassifier.py)
2. Bayes CNN: Bayes by Backprop, local reparameterization trick, variational dropout
3. SVM, KNN, Random Forest, and MLP with hyperparameter tuning (sklearn, sklearnClassifier.py)
4. CNN regressor: zero_padding enlarging mode in Resize(), target variable normalization techniques (sklearn)

## CNN (Modified LeNet)

![CNN loss][CNN_loss]

![CNN precision-recall curve][CNN_pr]

![CNN confusion matrix][CNN_cm]

[CNN_loss]: https://github.com/psychogeekir/ImageClassifier/blob/master/TLClassifier_result/compare_runningloss.png "CNN Loss"
[CNN_pr]: https://github.com/psychogeekir/ImageClassifier/blob/master/TLClassifier_result/nn_PR.png "CNN precision-recall curve"
[CNN_cm]: https://github.com/psychogeekir/ImageClassifier/blob/master/TLClassifier_result/nn_confusion_matrix.png "CNN confusion matrix"


## Bayes CNN (LeNet)

![BCNN loss][BCNN loss]
![BCNN pr][BCNN pr]
![BCNN cm][BCNN cm]

[BCNN loss]: https://github.com/psychogeekir/ImageClassifier/blob/master/BNNClassifier_result/compare_runningloss.png "BCNN Loss"
[BCNN pr]: https://github.com/psychogeekir/ImageClassifier/blob/master/BNNClassifier_result/nn_PR.png "BCNN pr"
[BCNN cm]: https://github.com/psychogeekir/ImageClassifier/blob/master/BNNClassifier_result/nn_confusion_matrix.png "BCNN confusion matrix"

## SVM

![SVM precision-recall curve][SVM_pr]

[SVM_pr]: https://github.com/psychogeekir/ImageClassifier/blob/master/SVC_result/SVC_PR.png "SVM precision-recall curve"


## CNN regressor

![CNN regressor loss][CNN regressor loss]

![CNN regressor r2][CNN regressor r2]

[CNN regressor loss]: https://github.com/psychogeekir/ImageClassifier/blob/master/TLRegressor_result/compare_runningloss.png "CNN regression loss"
[CNN regressor r2]: https://github.com/psychogeekir/ImageClassifier/blob/master/TLRegressor_result/nn_scatter.png "CNN true v.s. prediction"
