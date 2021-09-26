import cv2
from cv2 import dnn_superres

import sys

# check number of input arguments
if len(sys.argv) < 3:
	print('Too few arguments')
	exit()

modelIndex = int(sys.argv[1])
inputFile = sys.argv[2]

prependFolderInput = './input/'
prependFolderOutput = './output/'
prependFolderModels = './models/'
models = [
	{
		'model':'edsr',
		'fileName':'EDSR_x3.pb',
		'scale': 3
	},
        {
                'model':'edsr',
                'fileName':'EDSR_x4.pb',
                'scale': 4
        },
        {
                'model':'espcn',
                'fileName':'ESPCN_x4.pb',
                'scale': 4
        },
        {
                'model':'fsrcnn',
                'fileName':'FSRCNN-small_x4.pb',
                'scale': 4
        },
        {
                'model':'fsrcnn',
                'fileName':'FSRCNN_x4.pb',
                'scale': 4
        },
        {
                'model':'fsrcnn',
                'fileName':'FSRCNN_x2.pb',
                'scale': 2
        },
        {
                'model':'lapsrn',
                'fileName':'LapSRN_x8.pb',
                'scale': 8
        },
	{
                'model':'lapsrn',
                'fileName':'LapSRN_x4.pb',
                'scale': 4
        },
	{
                'model':'lapsrn',
                'fileName':'LapSRN_x2.pb',
                'scale': 2
        }
]

print(inputFile)

# Create an SR object
sr = dnn_superres.DnnSuperResImpl_create()

# Read image
image = cv2.imread(prependFolderInput + inputFile)

# Read the desired model
path = prependFolderModels + models[modelIndex]['fileName']
sr.readModel(path)

# Set the desired model and scale to get correct pre- and post-processing
sr.setModel(models[modelIndex]['model'], models[modelIndex]['scale'])

# Upscale the image
result = sr.upsample(image)


# Save the image
cv2.imwrite(prependFolderOutput + inputFile[:-4] + "-" + models[modelIndex]['fileName'][:-3] + ".png", result)
