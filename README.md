
#To get data:
wget -O pi0.pkl https://www.dropbox.com/s/hrdhr5o1khtclmy/pi0.pkl?dl=0
python utils/manual_split.py #This will split the physics data into training and testing

#To train a model:
python train_nflow.py

#To sample from trained model
python gen_NF_samples.py #Note: you need to enter into the source of this file and point to the location of a trained model dict

#To evaluate trained model (generate plots, etc)
python eval_flow.py #Note: you have the option to produce several kinds of plots in the source
