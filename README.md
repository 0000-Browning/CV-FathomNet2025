## CV-FathomNet2025

This is an entry in to the 2025 FATHOMNET Competition hosted on Kaggle.com (https://www.kaggle.com/competitions/fathomnet-2025/submissions). Training code for each model can be found in the folder labeled with the  same name as that model. Note that the baseline ("base") is a fine-tuned ResNet50 model.


# Run
1. Make a new python environment:
2. pip install -r requirements.txt
 -> Note to team: get a list of labels so this isn't required
3. Download the model of interest using the following link (ResNet-50 and Swin models supported): https://drive.google.com/drive/folders/1WHUaot542uYXNrnH14CUUGO3rxGdrNaS?usp=sharing
4. Put the model in samples/models
5. Find the test script for the model in the samples folder.
6. Run the test script:
$ python resnet_test.py --samples 5 --data data/ --output output/
7. The output csv will be generated in the output folder and the code will display a few plots

# Training set (Not supported directly, but if you want to run it here's how)
1. Download the data using $ python download.py dataset_train.json train/ [-n NUM_workers]
NOTE: this will take like 20 minutes
2. Run inference on the data in the same was as above (though yo

