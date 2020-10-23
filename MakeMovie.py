#!/usr/bin/env python

"""Main.py: Description."""

import numpy as np
from PIL import Image
import Parameters as p
import glob

entries = glob.glob("./Data/*.ppm")
length = len(entries)

writename = p.writename

files = []
a = np.arange(0, length, 1)
for value in a:
    entry = './Data/' + str(writename) + '.' + str(value).zfill(9) + '.ppm'
    files.append(Image.open(entry))
files[0].save('./Animations/' + str(writename) + '.gif',
              save_all=True, append_images=files[1:],
              optimize=False, duration=90, loop=0)
