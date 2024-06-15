Hello!
Please clone the git repo and use npm start to run the nodejs and express code and wait a few seconds before the images load in. They are saved in the images/ folder. 
The info.py file outputs a csv file with the scores of the downloaded images and renames the downloaded images as 1.jpg till 50.jpg. To run this use python info.py in the console.

To obtain the weights matrix:
Run the testing.py file under BAID folder (cloned from https://github.com/Dreemurr-T/BAID). 

The model.py file in the BAID folder was modified at lines 61 and 94 to include map_location="cpu" and the model_best.pth and epoch_99.pth files saved under BAID and ResNet_Pretrain subfolders, respectively.
