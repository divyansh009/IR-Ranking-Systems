MANUAL FOR EXECUTING THE ASSIGNMENT:
-----------------------------------------
1. Running Instructions:

*This assignment consists of a zip file named "21111027-assignment-1".

*Unzip this file to get 2 zip files named "21111027-ir-systems.zip" and "21111027-qrel.zip". We also have 1 makefile and readme file there.

*Open the terminal in linux operating system and set the path of terminal where the makefile lies.

*Type "make run" in the terminal.

*It will automatically unzip files and run the python code. We'll get the output files of 3 models in the folder named "Output_files".
.........................
2. Dependencies:

*Make sure python nltk library is installed. 
To install type the following command:
sudo apt install python3-nltk

*Stopwords from nltk must be installed too.
To install type the following command:
python3 -m nltk.downloader stopwords

*punkt from nltk must be installed too for tokenization.
To install type the following command:
python3 -m nltk.downloader punkt
.........................

Note:

*Preprocessing and posting list generator file is provided as ipynb files at location "All/Codes". Run them through jupyter notebook. It takes around 40-50 mins to execute the preprocessing file. Generated posting list are stored as pickle files that helps to skip this task.

*To run DataPreprocessing file, the corpus must be present at same location as the .ipynb file.
..............................

# Download a pickle file from this link and add it to 21111027-ir-systems\All\Saved:
https://iitk-my.sharepoint.com/:u:/g/personal/dbisht21_iitk_ac_in/Eas3rSsb1iVKu-Pukb7Z-TYBPcFBjq4H-dk489uR_myhPg?e=vWJF23
