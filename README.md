# ImageClassifier

- original image size (1 x 10 x 10), it is symmetric, use only the right-top quarter 
- 0: success, 1: failure (based on the top 20 quantile value of bandgap)
- imbalanced data (80/20)

## Models: 
1. CNN with data augmentation (i.e., random flip and random rotation) and imbalanced data sampler(pytorch, TLClassifier.py)
2. SVM, KNN, Random Forest, and MLP with hyperparameter tuning (sklearn, sklearnClassifier.py)
3. CNN regressor

## CNN (Modified LeNet)

![CNN loss][CNN_loss]

![CNN precision-recall curve][CNN_pr]

![CNN confusion matrix][CNN_cm]

[CNN_loss]: https://github.com/psychogeekir/ImageClassifier/blob/master/TLClassifier_result/compare_runningloss.png "CNN Loss"
[CNN_pr]: https://github.com/psychogeekir/ImageClassifier/blob/master/TLClassifier_result/nn_PR.png "CNN precision-recall curve"
[CNN_cm]: https://github.com/psychogeekir/ImageClassifier/blob/master/TLClassifier_result/nn_confusion_matrix.png "CNN confusion matrix"


## SVM

![SVM precision-recall curve][SVM_pr]

[SVM_pr]: https://github.com/psychogeekir/ImageClassifier/blob/master/SVC_result/SVC_PR.png "SVM precision-recall curve"


## CNN regressor


