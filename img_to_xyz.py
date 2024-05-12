from PIL import Image
import random

img = Image.open('redweed.jpg')
size = w, h = img.size
data = img.load()

array = []
coordsx = []
coordsy = []
'''
for x in range(0, w):
    for y in range(0, h):
        r, g, b = data[x, y]
        array.append((x, y, r, g, b))
        coordsx.append(x)
        coordsy.append(y)

randx = random.choice(coordsx)
randy = random.choice(coordsy)
'''
relative = []
coords = []

for x in range(0, w):
    for y in range(0, h):
        r, g, b = data[x, y]
        if 255 >= r >= 240 and 44 >= g >= 0 and 38 >= b >= 0:
            coords.append((x, y))

x1 = coords[0][0]
y1 = coords[0][1]

'''
for coord in coords:
    relative.append((coord[0]-x1, coord[1]-y1))
'''

with open('xy_col.data', 'w') as var:
    for arr in coords:
        var.write(str(arr).replace("'", '').replace(",", ' ').replace("(", '').replace(")", '')+'\n')