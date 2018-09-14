# -*- coding: utf-8 -*-
"""
Spyder Editor

Created on Sun Sep  9 20:31:59 2018
@author: Pavan Akula

References
- https://stackoverflow.com/questions/21790078/python-get-number-of-columns-from-csv-file?rq=1
- https://stackoverflow.com/questions/24398044/downloading-a-lot-of-files-using-python

"""

import os #Checks if directory and file existence
import urllib.request as request #Downloads files from url
from multiprocessing import Pool #multithreading to download multiple files
import csv #checks contents of the file

#Functions
def download_file(url):
    """
    Download the csv files from the website. In this case we are downloading files from GitHub
    """
    print('Downloading : {0:s}'.format(url))
    #Get file name from url
    file_name = str(url.split('/')[-1])
    file_path = "data/{0:s}".format(file_name)
    
    #If file already exists delete and re-download
    if os.path.exists(file_path):
        os.remove(file_path)
    
    #Download the file
    request.urlretrieve(url, file_path)
    
    print('Download Complete : {0:s}'.format(url))
    return 1


def directory_check(path):
    """
    Check data directory if not found, create new directory
    """
    if not os.path.isdir(path):
        print('Directory [' + path + '] does not exist... Creating directory [' + path + ']')
        os.system('mkdir ' + path)

def no_obs(path):
    """
    Check the csv file contents. Number of variables and observations.
    """
    obs = 0
    cols = 0
    with open(path, 'r') as f:
        csvlines = csv.reader(f, delimiter=',')   
        
        #Iterate the lines in CSV file using enumerate function
        for lineNum, line in enumerate(csvlines):
            #First line variable names
            obs = lineNum
            if lineNum == 0:
                cols = len(line) #counts number of variables in the file
            else:
                lineCols = len(line) 
                #Check if each observation has same number of variables
                #If does not match raise an error
                if cols != lineCols:
                    print("Line %s has less values in file %s" % (lineNum+1, path))

    print("Number of Variables: %s, Observations: %s in file: %s" %(cols, obs, path))


files = ['train.csv', 'test.csv'] #Create list of file names
urls = ["https://raw.githubusercontent.com/akulapa/Data622-HW2/master/{0:s}".format(f) for f in files] #Create list of URLs
paths = ["data/{0:s}".format(f) for f in files] #Create list of download path

#Check if data folder exists
directory_check('data')

#Download files
pool = Pool()
pool.map(download_file, urls)
pool.close()

#Check file contents
pool = Pool()
pool.map(no_obs, paths)
pool.close()
