This repository covers the evaluation and design work for my MSc Data Science project "End-to-End Deep Learning Architecture for Chronic Kidney Disease Prediction and Risk Assessment." With an emphasis on CNN-based architectures, the project seeks to enhance early CKD diagnosis and risk assessment by utilizing machine learning and deep learning techniques.
Details of the Dataset:
The UCI Machine Learning Repository provided the dataset for this study. It consists of 400 patient records with 24 clinical characteristics, such as clinical signs, outcomes of tests, and demographic data. Since the dataset is accessible to everyone under the Creative Commons Attribution 4.0 license and excludes any personal information, ethical standards have been respected.
Tools and Technologies:
Python 3 is the programming language.
Libraries used : Pandas, Matplotlib, Scikit-learn, TensorFlow, Keras, and NumPy
Frameworks: Colab by Google
Models:
The following steps form the methodology for this project:
Preprocessing of Data:
Utilizing statistical methods to deal with missing values (e.g., mode imputation for categorical data, mean imputation for numerical data).
To guarantee continuous feature scaling, the dataset should be standardized.
Numerical illustrations of categorical variables.
reducing class imbalance and enhancing the model's ability to recognize minority-class instances (CKD patients) by using the Synthetic Minority Oversampling Technique (SMOTE).
Selection of Features:
Recursive Feature Elimination (RFE) is a technique to simplify the dataset to identify the most important features for prediction.
Development of Models:
Three models are currently implemented and taught:
due to the ease of use and understanding, the Decision Tree (DT) model is frequently utilized as a baseline.
Applied for its instance-based learning technique, k-Nearest Neighbors (kNN) performs smoothly with balanced data.
The 12-layer design of convolutional neural networks (CNNs) includes batch normalization along with dropout layers to improve generalization and avoid overfitting.
enhancing each strategy hyperparameters to boost efficiency.
Evaluation of Performance:
Evaluating the models using metrics like:
Accuracy: Represents the number of cases that were accurately anticipated.
Precision: Assesses the extent to which the model reduces false positives.
The ability of an individual for precise determination of true positives (CKD cases) can be determined by recall.
For a fair review, the F1 Score provides a harmonic mean of recall and precision.
analyzing the models' performance on a hold-out test set to try to discover the best solution.
Analysis and Visualization:
examining the ROC curves and confusion matrices helps to understand the models' sorting skills.
understanding the contributions of various attributes to CKD prediction via visualizing feature importance.
Summary of the Results:
The best results have been achieved by the Convolutional Neural Network (CNN):
Accuracy: 97.50%
Precision: 98.41%
Recall: 97.42%
F1 Score: 97.33%
Contributors
Namratha Reddy Donthi, a University of Hertfordshire MSc Data Science 
studentIn charge: Luigi Alfonsi
Recognitions
A few people make this project possible:
Machine Learning Repository from UCI
Open-source libraries, which include Keras and TensorFlow
Assistance from Luigi Alfonsi, my supervisor
Permit:
The MIT License regulates the use of this repository. For extra details, see the LICENSE file.
Future Extent:
through multimodal data, especially lifestyle and genetic data, to enhance forecasts.
Research into lightweight models that are suitable for real-time CKD prediction in situations with limited resources.
with Explainable AI (XAI) techniques to enhance the clinical applicability and interpretability of models.




