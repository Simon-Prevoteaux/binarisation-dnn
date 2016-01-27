from PIL import Image
from os import walk
from os.path import relpath
from os.path import splitext

class DataSet(object):
    def __init__(self, folder):
        self.folder = folder
        self.imagespaths = {}

    def preload(self):
        """Finds all usable images in the path."""
        for (path, _, files) in walk(self.folder):
            path = relpath(path, self.folder)
            for filename in files:
                if path != '.':
                    filename = path + '/' + filename
                (name, _) = splitext(filename)
                if name in self.imagespaths:
                    print 'Warning: ', name, ' corresponds to several images in the dataset.'
                self.imagespaths[name] = self.folder + '/' + filename

    def images_names(self):
        """Returns a list of preloaded image names."""
        return self.imagespaths.keys()

    def open_image(self, imagename):
        """Opens an image.

        Args:
            imagename Name of the image, as returned by images_names (relative path without the extension).
        """
        try:
            return Image.open(self.imagespaths[imagename])
        except KeyError:
            print 'Unknown image name ', imagename
        except IOError:
            print 'I/O error while opening image ', imagespaths[imagename]
        return None

    def unload(self):
        """Unloads found images names."""
        self.imagespaths = {}

    def common_images(self, other):
        """Find common images between two data sets (=same size, same name)."""
        return self.imagespaths.viewkeys() & other.imagespaths.viewkeys()

