# nuorganics-spectroscopy
calibration models for nuorganics spectroscopy

# downloading the code
* download this repository

# data
place all Lab Report files and corresponding .TRM files into a folder named 'data'  
located at the root directory (spectroscopy/data)

# training the model
after getting the data, run the script `train_ammonia_model.py`.  
You should now have a folder spectroscopy/bin/model which contains the  
model file and the scores achieved by that model on the train and test sets

# using the model
* open up your terminal
* navigate to the folder where you downloaded the code using the `cd` command
* execute the command `python3 get_sample_ammonia_nitrogen.py -sp "path\to\samples\"`
* The results will be saved to the root directory where you have the program downloaded
