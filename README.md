## CV-FathomNet2025

This is an entry in to the 2025 FATHOMNET Competition hosted on Kaggle.com (https://www.kaggle.com/competitions/fathomnet-2025/submissions). Training code for each model can be found in the folder labeled with the  same name as that model. Note that the baseline ("base") is a fine-tuned ResNet50 model.


### Run

0. pip install -r requirements.txt
1. Download the data using $ python download.py dataset_test.json test/ [-n NUM_workers]
2. Download the data using $ python download.py dataset_train.json train/ [-n NUM_workers]
 -> Note to team: get a list of labels so this isn't required
3. Download the model of interest using the following link: https://drive.google.com/drive/folders/1WHUaot542uYXNrnH14CUUGO3rxGdrNaS?usp=sharing
4. Find the test script for that model (it will be in the corresponding folder for that model, or in the samples folder (NOT YET MADE))
5. Run the test script to fill in the labels in train/annotations.csv. The results will also be plotted (IMPLEMENT).



