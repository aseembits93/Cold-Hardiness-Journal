# Setup

* Set up conda environment with required packages in requirements.txt ```pip install -r requirements.txt```
* Install pytorch using ```conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch```

* Get models from [HERE](https://drive.google.com/drive/folders/19XEh8unC0wuQmslc3mTXAcley2-Bkgvb?usp=sharing)

* Download data and put in ./data/valid [https://github.com/AgAIDInstitute/Frost_Mitigation_Datasets/tree/master/ColdHardiness/Grapes/Processed/WashingtonState/Prosser/Python] [Note: this needs to be changed to pull directly from GitHub, but has not been done (yet) to make sure data is consistent across experiments] 

# Running the model

* Use this command to train the models
```
python main.py --experiment {single, mtl, concat_embedding}
```
* single = Single Task Learning Model
* mtl = MultiHead Model
* concat_embedding = Concatenation Embedding Model

Have a look at 'standalone_prediction.py' to get an idea about model deployment. It is set up with a default evaluation dataset and default pretrained model which can easily be changed.
