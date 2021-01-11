from tqdm import tqdm
import pandas as pd
import sys
import os

#location of all the train & test image folder (train_images & test_images), it will be used to fetch the names of the images
sampledir = 'drive/MyDrive/Plants/Sample/PlantDoc-Object-Detection-Dataset'

train        = pd.DataFrame()
data         = []
total_files  = os.listdir(os.path.join(sampledir,'train_images'))


for file_name in tqdm(total_files):
    if file_name[-3:] == 'jpg' or file_name[-3:] == 'png' or file_name[-3:] == 'JPG' or file_name[-3:] == 'PNG' :
        xml_file  = open(str(sampledir) + '/train_images/' + file_name[:-4] + '.xml' )       #replace sampledir with the directory of xml files
        etree = ET.parse(xml_file).getroot()
        for elm in etree.iter('name') : 
            name = str(elm.text)
            for elm in etree.iter('bndbox') : 
                tmp    = ''
                xmin   = int(elm[0].text)
                ymin   = int(elm[1].text)
                xmax   = int(elm[2].text)
                ymax   = int(elm[3].text)
                tmp    = str(sampledir) +'/train_images'+ '/' + str(file_name) + ',' + str(xmin) + ',' + str(ymin)+ ',' + str(xmax)+ ',' + str(ymax) + ',' + str(name.replace(' ', '-'))
                data.append(tmp)

train['format'] = data
train.head()


data.to_csv('annotted.txt',index=False)