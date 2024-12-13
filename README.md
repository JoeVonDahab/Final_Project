# CDD203 Final Project: hERG Blocker Prediction

This Colab notebook predicts whether a molecule is a hERG blocker, which is crucial for assessing cardiac toxicity during drug discovery. It uses machine learning models trained on SMILES representations of chemical compounds.

## Features

- **Preprocessing**: Validate SMILES strings and convert them to Morgan fingerprints.
- **Exploratory Data Analysis (EDA)**: Visualize and analyze dataset characteristics.
- **Machine Learning Models**:
  - Logistic Regression
  - Support Vector Machine (SVM)
  - Random Forest
  - Gradient Boosting
  - Neural Networks (with and without validation splitting)
- **Metrics**: Accuracy, precision, recall, F1-score, and AUC-ROC.
- **Prediction Functionality**: Predict hERG blocking activity for new SMILES strings.

## How to Use the Notebook

### 1. Open the Notebook on Colab

Click the link below to open the notebook in Google Colab:

(https://colab.research.google.com/github/JoeVonDahab/Final_Project/blob/main/CDD203_Final_Project.ipynb)

### 2. Install Dependencies

Execute the setup cells in the notebook to install required Python libraries:
```python
!pip install rdkit tensorflow keras

### 3. Run the Cells
Execute the cells in sequence to:

Preprocess the dataset
Train and evaluate models (CHose the model you would like to use, I recommend first two models)
Visualize performance metrics
Predict hERG blocker activity for new molecules 
### 4. Predict New Molecules
Use the provided prediction functions to test new SMILES strings:

you will find a cell that starts with:
# function to predict whether a molecule is a hERG blocker based on its SMILES
it has following code:

# function to predict whether a molecule is a hERG blocker based on its SMILES
def predict_blocker_forest(smiles_string):
    # create fingerprint given SMILES str
    fingerprint = generate_fingerprints(smiles_string)
    fingerprint_df = pd.DataFrame([fingerprint])
    # use the random forest model from earlier to predict
    prediction = rf_model.predict(fingerprint_df)[0]
    return prediction
# random example i use from the dataset to see if its working
new_smiles = "CC(=O)Oc1ccccc1C(=O)O"
prediction = predict_blocker_forest(new_smiles)
# 1 = blocker, 0 = non-blocker
print(f"Prediction for {new_smiles}: {prediction}")
just change the smile string to the one you would like to test

Authors
Youssef Abo-Dahab as part of final project for ML course.
Credit to my team Hanson Huang, Jackoson Sands, Su Oner

Happy Holiday





