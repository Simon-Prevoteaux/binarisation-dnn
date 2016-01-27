import cv2
from os import walk
from os.path import relpath
from os.path import splitext

class DataSet(object):
    def __init__(self, folder):
        self.folder = folder
        self.imagespaths = {}

    def preload(self):
        for (path, _, files) in walk(self.folder):
            path = relpath(path, self.folder)
            for filename in files:
                if path != '.':
                    filename = path + '/' + filename
                (name, _) = splitext(filename)
                if name in self.imagespaths:
                    print 'Warning: ', name, ' corresponds to several images in the dataset.'
                self.imagespaths[name] = self.folder + '/' + filename

    def imagesnames(self):
        return self.imagespaths.keys()

    def loadimage(self, imagename):
        try:
            return cv2.imread(self.imagespaths[imagename], cv2.IMREAD_GRAYSCALE)
        except KeyError:
            print 'Unknown image name ', imagename
        return None

    def unload(self):
        self.imagespaths = {}

