# Heart Disease Prediction using PyTorch Neural Network
## Lay Introduction
Heart disease remains one of the leading causes of death worldwide, making early detection vital for prevention and treatment. This project uses data and artificial intelligence to help predict whether a person is likely to have heart disease based on clinical measurements. Using patient data from the well-known Cleveland Heart Disease dataset, I trained a computer model to recognize patterns linked to heart disease. The dataset includes information such as age, blood pressure, cholesterol, and test results like ECG readings and exercise responses that doctors use and collect. By learning from these factors, the model can estimate a person’s risk of heart disease and even assign a “risk score” showing how likely the condition is.

The model achieved high accuracy (about 88%) and identified key predictors like blocked arteries, chest pain type, and maximum heart rate. While the dataset is small and mostly male, the results show how machine learning can help support decision making in a medical setting. Implications of this work include a tool for use by clinicians as an assistant, the demonstration of techniques that prevent models from memorizing data and instead focus on learning patterns, and improving identification of at-risk patients using data collected in the past. 

## Project Overview
This capstone project implements a complete machine learning pipeline to predict the presence of heart disease in patients using clinical data sourced from the UCI Machine Learning Repository, Cleveland Heart Dataset. I focused on using a neural network with literature-supported architecture and properties in order to predict whether a patient had heart disease or not given clinical data, ranging from resting blood pressure to age. I was also able to produce risk scores due to the probabilistic nature of the model I chose. Finally, I validated the relationships between these patient attributes and heart disease risk by performing statistical analyses, visualizations, and repetition using other models. The goal of this work was to identify key clinical risk factors for heart disease and build a reliable predictive model for clinicians to use.

## 1. Background
Cardiovascular disease is a leading cause of mortality worldwide, particularly in the United States. Early and accurate detection is crucial for effective intervention and treatment. According to Cleveland Clinic, heart disease, a subset of cardiovascular diseases, involves issues such as narrowing of blood vessels, congenital heart and vessel problems, and arrhythmias. These conditions can collectively stress the cardiovascular system and lead to heart failure, increased risk of heart attacks, and chronic pain. High blood pressure, lack of physical activity, genetics, and cholesterol levels can contribute to the development of heart disease. Clinical decision-making can be enhanced by the adoption of data-driven, AI-integrated solutions that take advantage of ample amounts of patient data. This project specifically harnesses the Cleveland Heart Disease dataset from the UCI Machine Learning Repository, a well-known benchmark dataset in medical informatics, which contains 13 clinical features from 303 patients, of which 297 had complete datasets. These features are listed below, and include demographic history, medical history, and diagnostic test results, such as visualizations of major blood vessels. In the future, this simulation of machine learning in cardiovascular diagnostics aims to become a reality.

### Feature Set (all sourced from the Cleveland Heart Dataset Documentation)

#### Categorical Variables
- **Sex** - Binary: males have higher risk.
- **Chest Pain Type (cp)** - Ordinal: Recorded as typical, atypical, non-anginal, or asymptomatic; anginal pain reflects ischemia, or loss/reduction in blood flow to heart. Non-anginal means chest pain due to other causes. Can be typical or atypical depending on patient presentation and symptoms.
- **Exercise-Induced Angina (exang)** - Binary: Whether chest pain occurs on physical exertion. Pain on exertion indicates limited coronary flow, which has a strong correlation with heart disease. Yes or no.
- **Fasting Blood Sugar > 120 mg/dl (fbs)** - Binary: Whether fasting blood sugar exceeds this threshold. If it does, it is an indicator of diabetes; chronic hyperglycemia damages endothelium, and can lead to injury to arteries/heart.
- **Resting ECG (restecg)** - Ordinal: Recorded as either normal, ST-T abnormalities, or LV hypertrophy; abnormalities suggest ischemia or structural heart disease.
- **Slope of Peak Exercise ST Segment (slope)** - Ordinal: Can be upsloping, flat, or downsloping; all indicators from ECG signals from patients that can indicate risk of heart attack due to arrhythmia/heart disease.
- **Number of Major Vessels Colored by Fluoroscopy (ca)** - Ordinal: Direct measure of blood vessel constriction. If either 1, 2, or 3 major vessels that can be visualized via this technique, indicating stenosis, a predictor of heart disease.
- **Thallium Stress Test Result (thal)** - Ordinal: Uptake of signal (perfusion) into cardiac muscle. Is either 3 (normal), 6 (fixed defect), or 7 (reversible defect). Measured to show heart flow defects on a scan; indicates heart damage, stress, or reduction in blood flow to heart.

#### Continuous Variables
- **Age** - Continuous: positively correlated with disease as arteries stiffen and risk factor exposure grows.
- **Resting Blood Pressure (trestbps)** - Continuous: Hypertension (high BP) injures the arteries over time and can stress the heart.
- **Serum Cholesterol (chol)** - Continuous: Cholesterol levels in patient serum. Elevated levels indicate atherosclerosis, or plaque buildup that can again stress the vessels and heart.
- **Maximum Heart Rate Achieved (thalach)** - Continuous: Lower peak indicates impaired function of the heart.
- **ST Depression from Exercise (oldpeak)** - Continuous: Exercise-induced depression reflects ischemia, or reduction in blood flow to heart, upon exertion.

#### Target/Label Variable
- **Heart Disease Diagnosis (target)** - Binary (after processing)
  - Original: 0 = no disease, 1-4 = varying degrees of disease severity
  - Processed: 0 = no disease, 1 = presence of disease

## 2. Motivation
The primary motivation is to demonstrate the application of deep learning in a pressing, high-impact region of medicine, cardiovascular disease. While many studies use traditional models like Logistic Regression for this task, this project explores whether a Neural Network can offer superior performance. I also wanted to test whether a neural network with early stopping criteria could avoid overfitting and work on simply 300 or so data points. Furthermore, by conducting a feature importance analysis and statistical analysis, this project aims to showcase the features most important for strong and robust diagnoses of heart disease, including risk factors.

## 3. Summary
### Note on Project Version: This project has a 3-5 page condensed version of the heart disease prediction project for the submission criteria, and a full version with the full analysis. Unlike the full implementation, the 3-5 page version utilizes a simplified training approach with a fixed 30-epoch run on training data without early stopping or validation set separation. The dataset is split 80-20 between training and testing without an intermediate validation partition. Key analyses included in this version are: categorical feature visualizations (bar plots) – corresponding to Figure 2 in the full version – neural network training curves (loss and accuracy), and final model evaluation metrics (confusion matrix, ROC curve, and precision-recall curve), corresponding to Figure 4 from the full version. The model architecture remains consistent with literature-inspired design featuring batch normalization and dropout regularization, but training is conducted to a predetermined epoch count rather than using early stopping criteria.
### Data Processing
The project begins by loading the Cleveland dataset, handling missing values in the 'ca' and 'thal' columns by removing affected rows, resulting in 297 clean samples. The target variable, which originally indicated disease severity (0-4), was binarized (0 for no disease, 1 for disease). The data was then split into training (65%), validation (15%), and test (20%) sets, with stratification to preserve the target distribution. All features were standardized using `StandardScaler` as was done in previous assignments for the class, and allows for efficient model training.
### Neural Network
A custom PyTorch Neural Network was implemented with two hidden layers (32 and 16 neurons, although running the model was done with 64 and 32 neurons with further testing as this improved accuracy considerably), Batch Normalization, Dropout for regularization, and a Sigmoid output layer. The model was trained with early stopping adapted from Keras to prevent overfitting. The model was trained each epoch on batches of training data and validated on batches of validation data to identify improvements in validation loss/accuracy – if a minimum improvement is not attained, the model training was halted at the given epoch (which in practice occurred between 21-25 epochs). The model was then tested on the held-out testing dataset and performance was analyzed.

### Figures

- **Fig 1a-1h (Exploratory Data Analysis):** This series of plots provides a first-part analysis of the data.
  - **Fig 1a:** Pie chart showing distribution of heart disease prevalence in the patient sample, showing a relatively balanced dataset (54% No Disease, 46% Disease).
  - **Fig 1b:** Distribution of patient population in bins of 10, showing that the most common age bin is 55-65 years old. Very few patients are under 35, expected for a disease with a high correlation with aging.
  - **Fig 1c:** Proportion of patients in each age bin that have heart disease, showing that 64% of the patient population sampled between 55-65 years old has heart disease.
  - **Fig 1d, 1e, 1g:** Boxplots show the distribution of cholesterol levels, maximum heart rate, resting blood pressure, and ST depression compared between patients with heart disease and patients without. Although significance testing was not done here, patients without heart disease typically attain higher maximum heart rates, but have lower ST depression values, falling in line with documentation. Cholesterol distribution and Resting BP are roughly aligned between the two disease state groups.
  - **Fig 1h:** Pie chart with sex distribution of patient population showing a significant overrepresentation of male patients (68%).

- **Fig 2a-2f (Categorical Feature Analysis):** These bar plots illustrate the relationship between categorical features and disease prevalence.
  - **Fig 2a:** High blood sugar prevalence (0 for normal blood sugar, 1 for high) compared between patients with and without heart disease, showing similar distributions.
  - **Fig 2b:** Resting ECG results compared between disease state groups. Patients without heart disease were more likely to have normal ECG results, while patients with heart disease had higher rates of left ventricular hypertrophy.
  - **Fig 2c:** Chest pain during exercise – patients without heart disease had significantly higher rates of a "yes", or 1.0, for this feature.
  - **Fig 2d:** Types of chest pain compared between disease state groups. Patients without heart disease surprisingly had higher rates of anginal and non-anginal pain, while patients with heart disease were more likely to be asymptomatic.
  - **Fig 2e:** Slope of ST segment after exercise, indicative of heart recover-ability, showing patients with heart disease having more flat/downsloping ST segments post-exertion, with controls having more upsloping.
  - **Fig 2f:** Number of blocked/stenosis-implicated major blood vessels between disease state groups, showing patients with heart disease having significantly higher rates of 1, 2, or 3 major blocked, visualized vessels. Most controls had no vessels blocked.

- **Statistical Testing:** T-tests with Bonferroni correction identified `age` - Age, `sex` - Sex, `cp` - Chest Pain Type, `thalach` - Maximum Heart Rate Achieved, `exang` - Exercise-Induced Angina, `oldpeak` - ST Depression Induced by Exercise, `slope` - Slope of Peak Exercise ST Segment, `ca` - Number of Major Vessels Colored by Fluoroscopy, and `thal` - Thallium Stress Test Result, as statistically significant predictors of heart disease. The mean, median, standard deviation, and other statistical metrics were shown for these variables/predictors.

- **Neural Network Model:** A custom PyTorch Neural Network was implemented with two hidden layers (64 and 32 neurons), Batch Normalization, Dropout for regularization, and a Sigmoid output layer. The model was trained with early stopping to prevent overfitting.

- **Fig 3a & 3b (Training Curves):** These plots show the model's learning process. Training and validation loss decreased steadily, and accuracy increased, with early stopping triggered at epoch 24 due to early-stopping criteria/regularization.

- **Model Evaluation (Fig 4a-4c):** On the test set, the Neural Network achieved strong performance, specifically an accuracy of 88.3% and AUC-ROC of 0.955, corroborated by figures 4b and 4c showing ROC curve and precision-recall curve. Confusion matrices (4a) show effective classification of both heart disease positive and negative groups, with slightly better performance predicting patients without heart disease.

- **Feature Importance (Fig 5):** Using a permutation-based method, the top 5 most important features were identified as: number of major vessels visualized via fluorophores (`ca`), chest pain type (`cp`), thalassemia (`thal`), maximum heart rate achieved (`thalach`), and sex (`sex`). Positive values mean that when randomizing the value of these features, the accuracy was reduced, on average, over four trials per feature. The following tables showcase these top features, the average accuracy drop, and features reducing accuracy – in this case, the only variable for whom randomization increased model accuracy was fasting blood sugar exceeding 120 mg/dl, which fits with the lack of any noticeable difference in the distribution of this feature between the heart disease positive and negative groups.

- **Model Comparison (Fig 6a-6f):** The Neural Network was compared against Logistic Regression, Random Forest, and SVM. It achieved the best scores in Accuracy, Precision, Recall, and F1-score, while SVM had a marginally better AUC-ROC (0.959). This demonstrates the neural networks strong performance despite having a limited dataset, particularly with the help of early stopping criteria and a relatively small-scale structure with reduced neurons per layer. SVMs are also highly usable in this situation due to the smaller dataset, binary classification need, and relatively low number of features used for classification. The caption below the figure does say that SVM overperforms the neural network, as multiple runs showed higher AUCs and more consistent accuracy scores for the SVM model.

- **Risk Stratification (Fig 7a & 7b):** Patients were categorized into Low, Medium, and High-risk groups based on predicted probability, but only for the test dataset with n = 60. Nearly 50% of the patients were at low-risk, while a third were high risk. The probability criteria were set to 0.3 and 0.7 for the three-category-differentiation to reflect the benefit of being safer and taking greater precaution. The distribution of probability values by risk category is also shown in Figure 7b. Finally, the output below the figures showcases the distribution of low, medium, and high risk by each age group, and the number of patients in each predicted risk group that actually have heart disease. For example, for age group 45-55, 0 out of 2 patients predicted as medium risk had heart disease, while 8 out of 8 patients predicted as high risk had heart disease. As age increases, the patients marked as medium risk are more and more likely to actually have heart disease. No patients in the test set were between 25 and 35 years old due to the low sample size from this age bin.

### Results and Analysis
The project successfully built a predictive model with 88.3% accuracy. The analysis confirmed well-known clinical risk factors, such as the number of blocked vessels, type of chest pain, and maximum heart rate. The Neural Network performed comparably or better than other models despite the smaller dataset. The risk stratification also shows that the model can be used to predict high-risk subgroups, which should be interpreted with respect to other variables such as age. The statistical tests also support the model output and identify well-known clinical factors such as high blood pressure, maximum heart rate, among others, as highly correlated with heart disease.

## 4. Implications and Limitations

### Implications
This project illustrates that even with a small dataset, by adapting a neural network to include early stopping criteria and incorporate concepts such as dropout and batch normalization, computational models can align with established clinical understanding of heart disease and serve as a tool for clinical decision-making. This model can be used as a screening assistant to help clinicians prioritize high risk patients, identify patients at risk in the future, and validate the importance of certain metrics with data.

### Limitations
1. **Small Dataset:** With only 297 samples after cleaning, the model is susceptible to overfitting, and the results may not generalize well to a broader population. The performance of all models should be validated on a larger dataset.
2. **Data Imbalance:** The dataset is skewed towards male patients (68%), which could introduce bias into the model, potentially reducing its accuracy for female patients. According to Cleveland Clinic, female presentation of heart disease and acute effects of heart disease significantly differs from males, which may limit model capability.
3. **Simplicity of Model:** The Neural Network architecture is relatively simple to avoid overfitting and adjust for small datasets. The architecture is a simplified and modified version of literature-backed architecture for disease prediction, and this may result in omission of critical architectural components relating to prediction.
4. **Correlation Constraint:** The analysis identifies associations but cannot prove causation between the features and heart disease.

## 5. Overall Message About the Data
The Cleveland Heart Disease dataset provides a valuable, though limited, snapshot of clinical factors associated with heart disease. The analysis robustly shows that non-invasive measures like chest pain type, exercise-induced angina, and ECG results are highly predictive. It also highlights that while complex models like Neural Networks can perform excellently, simpler models are also very effective on this scale of data. The key takeaway is that data-driven models can uncover and quantify known clinical relationships using datasets, altogether supporting clinicians as a predictive, reliable tool.

## 6. Full Citation List

1. **Data Source:** Detrano, R. (1989). International application of a new probability algorithm for the diagnosis of coronary artery disease. *American Journal of Cardiology, 64*, 304-310. UCI Machine Learning Repository: [Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease).

Dataset Link: Heart Disease - UCI Machine Learning Repository

2. **Methodological References**
   1. Neural Network Architectures in Biomedical Applications Srivastava, Prof. (Dr.) N., & Chatterjee, P. (2025). Achieving Superior Predictive Performance through Deep Neural Networks for Enhanced Cardiovascular Disease Detection. International Journal of Research Publication and Reviews, 6(6), 5964–5968. https://doi.org/10.55248/gengpi.6.0625.21113
   2. PyTorch Methods, EarlyStop Sourcing, etc.: Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Köpf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., & Bai, J. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. ArXiv.org. https://arxiv.org/abs/1912.01703

3. **Library References:**
   1. Matplotlib Hunter, J. D. (2007). "Matplotlib: A 2D Graphics Environment." Computing in Science & Engineering, 9(3), 90–95. Official Documentation: https://matplotlib.org/stable/contents.html
   2. Seaborn Used for statistical visualization. Official Documentation: https://seaborn.pydata.org/
   3. Pandas Used for data handling and preprocessing. Documentation at Official Documentation: https://pandas.pydata.org/docs/
