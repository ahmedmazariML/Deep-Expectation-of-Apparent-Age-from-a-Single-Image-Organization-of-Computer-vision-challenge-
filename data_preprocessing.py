import scipy.io
import pandas as pd
from datetime import date
import datetime
import re
import numpy as np
from scipy.misc import imread, imresize

def ordinalToDate(ordinal):
    return date.fromordinal(ordinal) + datetime.timedelta(days=ordinal % 1) - datetime.timedelta(days=366)


def removeNonAlphaNumericalCharacters(strToClean):
    return re.sub("\"|u|\[|\'|\]", lambda s: "", str(strToClean))


def binarize(age):
    return int((age > 30))

mat = scipy.io.loadmat('input/wiki_meta_data.mat')

dateOfBirths = mat['wiki']['dob'][0][0][0]
dateOfPhotoShoot = mat['wiki']['photo_taken'][0][0][0]
filePath = mat['wiki']['full_path'][0][0][0]
celebrityName = mat['wiki']['name'][0][0][0]
faceScore = mat['wiki']['face_score'][0][0][0]

wikiData = pd.DataFrame(data={

    'date_of_birth': dateOfBirths,
    'date_of_photo_shoot': dateOfPhotoShoot,
    'file_path': filePath,
    'faceScore': faceScore

})

wikiData['file_path'] = wikiData['file_path'].apply(removeNonAlphaNumericalCharacters)

wikiData['date_of_birth'] = wikiData['date_of_birth'].apply(lambda d: ordinalToDate(d))
wikiData['age'] = wikiData['date_of_photo_shoot'] - wikiData['date_of_birth'].apply(lambda d: d.year)

wikiData.replace([np.inf, -np.inf], np.nan)
wikiData.replace(np.inf, np.nan)

wikiData = wikiData.drop('date_of_birth', 1)
wikiData = wikiData.drop('date_of_photo_shoot', 1)
wikiData['label'] = wikiData['age'].apply(binarize)
cleanData = wikiData.replace([np.inf, -np.inf], np.nan).dropna()

# cleanData.to_csv('output/cleaned.csv')
# wikiData.to_csv('output/example.csv')

# print(cleanData.shape)
numberOfRows = cleanData.shape[0]
#
# print(cleanData['file_path'][:10])
# print(cleanData['label'][:10])
nFeature = 224 * 224 * 3


print("Build raw data set")
with open(r'test.txt', 'w') as f:

    for i in cleanData.index[:1000]:

        img = imread("input/wiki_crop/" + cleanData['file_path'][i])

        label = cleanData['label'][i]
        img = imresize(img, (224, 224))

        if (len(img.shape) == 2):
            img = img.reshape((224 * 224))
            img = np.hstack((img,img,img, label))
        else:
            img = img.reshape(nFeature)
            img = np.hstack((img, label))

        f.write(" ".join(np.char.mod('%d', img)) + "\n")

f.close()
