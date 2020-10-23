#!/usr/bin/env python

"""Main.py: Description."""

import numpy as np
from PIL import Image
import Parameters.py as p

writename = p.writename

files = []
a = np.arange(0, 486, 1)
for value in a:
    entry = './Data/' + str(p.writename) + '.tipsy.' + str(value).zfill(9)+'.ppm'
    files.append(Image.open(entry))
files[0].save('./Animations/ +' str(p.writename) + '.gif',
              save_all=True, append_images=files[1:],
              optimize=False, duration=90, loop=0)
