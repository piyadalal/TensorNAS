
import io, json
import sys
import time
import os
PYTHON_VERSION = sys.version_info[0]
if PYTHON_VERSION == 2:
    from urllib import urlretrieve
elif PYTHON_VERSION == 3:
    from urllib.request import urlretrieve


def download(self, tarDir=None, imgIds=[]):
    '''
    Download COCO images from mscoco.org server.
    :param tarDir (str): COCO results directory name
           imgIds (list): images to be downloaded
    :return:
    '''
    if tarDir is None:
        print('Please specify target directory')
        return -1
    if len(imgIds) == 0:
        imgs = self.imgs.values()
    else:
        imgs = self.loadImgs(imgIds)
    N = len(imgs)
    if not os.path.exists(tarDir):
        os.makedirs(tarDir)
    for i, img in enumerate(imgs):
        tic = time.time()
        fname = os.path.join(tarDir, img['file_name'])
        if not os.path.exists(fname):
            urlretrieve(img['coco_url'], fname)
        print('downloaded {}/{} images (t={:0.1f}s)'.format(i, N, time.time() - tic))

# get the image ids from annotations text file and download and store the images in a directory called dataset in demos :: differemt dorectory for person and non person images
"""Reading the Json File to get imgids"""




data = [json.loads(line) for line in open('./demos/annotations_person.json', 'r')]

