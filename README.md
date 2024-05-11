# Setup

* Set up a new conda environment ```conda create -n chjournal python=3.9```
* ```conda activate chjournal```
* Set up conda environment with required packages in requirements.txt ```pip install -r requirements.txt```
* Install pytorch using ```conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia```

# Running the model

* Use the following commands to train the baseline models
```
python main.py --experiment concat_embedding --name journal_concate
python main.py --experiment mtl --name journal_multih
python main.py --experiment single --name journal_single
python main.py --experiment ferguson --name ferguson
```
* Use the following commands to generate the main set of grape cold hardiness results. If you don't want to train the models from scratch, I've uploaded the pretrained models [here](https://oregonstate.box.com/s/981pith51vryyoe2ec5vmegjxj0kf5e2)
```
python main.py --experiment concat_embedding --name journal_concate --evaluation
python main.py --experiment mtl --name journal_multih --evaluation
python main.py --experiment single --name journal_single --evaluation
python main.py --experiment ferguson --name ferguson --evaluation
```

* a copy of the generated results is available in ```results/```