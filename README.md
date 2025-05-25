# Psychotherapy_Treatment_Dropout

This repository contains the implementation of a GNN approach to generate embeddings of medical features per patient. The architecture is trained on structured Electronic Health Records (EHR) data, specifically on psychosis patients in the WRAP program at Boston Medical Center (BMC). The goal of this project is to identify patients at risk of discontinuing psychotherapy.

## Dataset
Patient population: 2380.
Features includes:
- Demographic variables (age, sex, ethnicity, race, language, insurance type, etc.)
- Height, weight, pulse, BMI
- Psychiatric diagnoses (ICD-10 F0-F90)
- Antipsychotic medications (clozapine, risperidone, olanzapine, etc.)




## Outcome extraction
To process the treatment dropout as an outcome, we first defined the criteria for the following 3 classes:
- Active: Has a qualifying encounter at least every 6 month. 
- Re-engaged: If there exists a gap between encounters that is larger than 6 months, but comes back to regular visits.
- Dropped out: If there exists a gap greater than 6 months, and does not return back to regular visits. 

We have carefully filtered qualifying encounters by considering outpatient encounters that are: (1) in the correct psychiatric department, (2) completed, and (3) patients that have been in the system for at least 6 months.

The following multi-class distribution are shown:
- Active         39.36%
- Dropped Out      24.65%
- Re-engaged     35.99%

## Installation

```bash
git clone https://github.com/Khangxlei/Psychotherapy_Treatment_Dropout.git
cd Psychotherapy_Treatment_Dropout
pip install -r requirements.txt
```

## Train and inference imputation model

1. Run the Jupyter notebook imputation_train.ipynb
2. After training, we inferenced the trained model onto our dataset by running:
```bash
python imputation_inference.py
```

## Variables and outcome processing
1. First run:
```bash
python process_features.py
```

2. Then run:
```bash
python variables_processing.py
```
## GNN Training and Evaluation
After all preprocessing has been done, we can now run our neural network architecture to generate embeddings for our patients, which will then gets feed into standard ML predictive models (XGBoost, Random Forest, etc.) for predictions.

To do so, simply run all the cells of gnn_preprocess_3_classes.ipynb to also predict re-engagement. If we just want to predict 2 classes (dropout vs. not) then run the cells of gnn_preprocess_2_classes.ipynb. 
