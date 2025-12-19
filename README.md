<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Credit Card Default Prediction - ML Project</title>
    <style>
        body {
            font-family: Arial, Helvetica, sans-serif;
            line-height: 1.7;
            margin: 40px;
            background-color: #f9f9f9;
            color: #222;
        }
        h1 {
            text-align: center;
            color: #2c3e50;
        }
        h2 {
            color: #1f6feb;
            border-bottom: 2px solid #ddd;
            padding-bottom: 5px;
            margin-top: 40px;
        }
        h3 {
            color: #34495e;
            margin-top: 25px;
        }
        ul {
            margin-left: 20px;
        }
        li {
            margin-bottom: 8px;
        }
        .box {
            background: #ffffff;
            padding: 20px;
            border-left: 5px solid #1f6feb;
            margin: 20px 0;
        }
        .diagram {
            background: #eef3ff;
            padding: 15px;
            border-radius: 6px;
            font-family: monospace;
            white-space: pre-wrap;
        }
        .highlight {
            color: #d35400;
            font-weight: bold;
        }
        footer {
            text-align: center;
            margin-top: 60px;
            font-size: 14px;
            color: #555;
        }
    </style>
</head>
<body>

<h1>💳 Credit Card Default Prediction using Machine Learning</h1>

<div class="box">
    <p>
        This project focuses on predicting whether a customer will 
        <span class="highlight">default on a credit card payment</span>
        using multiple Machine Learning models.  
        The entire pipeline is built using <b>classes, objects, and functions</b> 
        to keep the code modular, readable, and production-ready.
    </p>
</div>

<h2>📌 Project Objective</h2>
<ul>
    <li>Analyze credit card customer data</li>
    <li>Clean and preprocess the dataset</li>
    <li>Train multiple ML models</li>
    <li>Compare model performance using metrics and curves</li>
    <li>Select the best model and deploy it as a web application</li>
</ul>

<h2>🧱 Project Architecture</h2>
<div class="diagram">
Raw Data
   ↓
Data Cleaning
   ↓
Missing Value Handling
   ↓
Outlier Treatment
   ↓
Feature Selection
   ↓
Data Balancing
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Best Model Selection
   ↓
Pickle File
   ↓
Web Deployment
</div>

<h2>🧹 Data Preprocessing</h2>

<h3>1️⃣ Null Value Check & Removal</h3>
<ul>
    <li>Checked the dataset for null values</li>
    <li>Removed rows containing null values to ensure clean data</li>
</ul>

<h3>2️⃣ Missing Value Handling</h3>
<ul>
    <li>Identified missing values in relevant features</li>
    <li>Handled missing values using <b>Random Sampling Imputation</b></li>
    <li>This method preserves the original data distribution</li>
</ul>

<h3>3️⃣ Outlier Detection & Handling</h3>
<ul>
    <li>Detected outliers using statistical techniques</li>
    <li>Applied variable transformation to reduce the effect of extreme values</li>
    <li>Improved model stability and accuracy</li>
</ul>

<h3>4️⃣ Feature Selection</h3>
<ul>
    <li>Removed irrelevant and less important columns</li>
    <li>Reduced dimensionality and overfitting</li>
    <li>Improved model performance and training speed</li>
</ul>

<h3>5️⃣ Data Balancing</h3>
<ul>
    <li>Observed class imbalance in the target variable</li>
    <li>Applied balancing techniques to equalize class distribution</li>
    <li>Ensured fair learning for all models</li>
</ul>

<h2>🤖 Machine Learning Models Used</h2>
<ul>
    <li>K-Nearest Neighbors (KNN)</li>
    <li>Naive Bayes</li>
    <li>Logistic Regression</li>
    <li>Decision Tree</li>
    <li>Random Forest</li>
    <li>AdaBoost</li>
</ul>

<h2>📊 Model Evaluation</h2>
<div class="box">
    <ul>
        <li>Train-Test split was applied</li>
        <li>Each model was trained on the same dataset</li>
        <li>Performance was evaluated using:</li>
        <ul>
            <li>Test Accuracy</li>
            <li>Classification Report (Precision, Recall, F1-score)</li>
            <li>ROC Curve</li>
            <li>AUC-ROC Score</li>
        </ul>
    </ul>
</div>

<h2>📈 ROC & AUC Curve Analysis</h2>
<div class="diagram">
True Positive Rate (TPR)
│
│        Logistic Regression
│        ───────────────────
│       /￣￣￣￣￣￣￣￣￣
│      /
│     /
│    /
│   /
│__/________________________ False Positive Rate (FPR)
</div>

<ul>
    <li>ROC curves of all models were compared</li>
    <li><b>Logistic Regression</b> showed the best AUC score</li>
    <li>Provided the best balance between sensitivity and specificity</li>
</ul>

<h2>🏆 Best Model Selection</h2>
<div class="box">
    <p>
        Based on accuracy, classification report, and ROC-AUC curve analysis,  
        <span class="highlight">Logistic Regression</span> was selected as the final model.
    </p>
</div>

<h2>💾 Model Saving</h2>
<ul>
    <li>The finalized Logistic Regression model was saved using <b>Pickle</b></li>
    <li>This allows easy reuse without retraining</li>
</ul>

<h2>🌐 Deployment</h2>
<ul>
    <li>Built a web application using the trained model</li>
    <li>Integrated the Pickle file with the backend</li>
    <li>Users can input data and get real-time predictions</li>
</ul>

<h2>✨ Key Highlights</h2>
<ul>
    <li>End-to-end Machine Learning pipeline</li>
    <li>Object-Oriented Programming (OOP) approach</li>
    <li>Multiple model comparison</li>
    <li>Strong evaluation using ROC & AUC</li>
    <li>Production-ready deployment</li>
</ul>

<h2>✅ Conclusion</h2>
<div class="box">
    <p>
        This project demonstrates a complete <b>end-to-end Machine Learning solution</b> for 
        <span class="highlight">Credit Card Default Prediction</span>, starting from raw data preprocessing 
        to final deployment as a web application.
    </p>

    <p>
        Proper data cleaning, missing value handling using random sampling, outlier treatment, feature selection,
        and data balancing played a crucial role in improving model reliability and performance.
    </p>

    <p>
        Multiple classification models were trained and evaluated using accuracy metrics, classification reports,
        and ROC–AUC curves. Among all models, <span class="highlight">Logistic Regression</span> achieved the best
        overall performance and was selected as the final model.
    </p>

    <p>
        The trained model was saved using a Pickle file and successfully deployed, enabling real-time predictions.
        This project highlights how Machine Learning models can be transformed into practical, real-world solutions.
    </p>
</div>

<footer>
    🚀 Credit Card Default Prediction Project | Machine Learning
</footer>

</body>
</html>
