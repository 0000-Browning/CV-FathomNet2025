## CV-FathomNet2025

This is an entry in to the 2025 FATHOMNET Competition hosted on Kaggle.com (https://www.kaggle.com/competitions/fathomnet-2025/submissions). Training code for each model can be found in the folder labeled with the same name as that model. Note that the baseline ("base") is a fine-tuned ResNet50 model.

# WARNING: CLONING THIS REPO WILL LIKELY TAKE AT LEAST FEW MINUTES (It took upwards of 20 minutes for some of us)

## Running sample code
1. Make a new python environment:

   a. python -m venv venv

   b. source venv/bin/activate
3. pip install -r requirements.txt
4. Download the model of interest using the following link (ResNet-50 SGD and Swin models supported by default):
   [Models Link](https://drive.google.com/drive/folders/1WHUaot542uYXNrnH14CUUGO3rxGdrNaS?usp=sharing)
5. Put the model in the models/ folder
6. Find the test script for the model in the samples folder.
7. cd into sample folder
8. Run the test script:
   (these are seprate models)

   $ python resnet_test.py

   $ python swin_test.py

   (there are command line arguments for both if you want to change input and output folders)
9. The output csv will be generated in the output/ folder and the code will display a few plots

## Running sample code on training set (Not supported directly, but if you want to run it here's how)

1. [FathomNet Data Download instructions](https://www.kaggle.com/competitions/fathomnet-2025) Download the data using $ python download.py dataset_train.json data/ [-n NUM_workers]
   NOTE: this will take like 20 minutes unless you use OSC resources
2. Then you can use the command line args to specify the paths to the input and output csvs and models.

## Running other models

1. All the other basic ResNet-50 models should run with resnet_test.py, you just have to specify the model path with --model [model_path]
2. BioCLIP is a pain to run with our setup since it requires some data manipulation to get the data in the correct format
4. Hierarchy test code is in the hierachy folder, but it wasn't tested directly so you may have to mess with it a bit to get results
