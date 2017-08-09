import PIL
from PIL import Image
import numpy as np
images = ['a.jpg', 'b.jpg', 'd.jpg', 'g.jpg']

for image in images:

    basewidth = 28
    img = Image.open('test_images/original/' + image)
    print np.array(img)
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
    img.save('test_images/resized/' + image)