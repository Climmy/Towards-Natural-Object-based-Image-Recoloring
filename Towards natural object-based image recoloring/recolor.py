from PIL import Image
from palette import *
from util import *
import random
from transfer import *
import time
from itertools import permutations
# from test import *
#color number in palette
num=[ 0,0,4,4,0,0,0,0,4,0,
      4,3,0,4,4,4,4,4,4,4,
      4,4,4,4,0,4,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,4,5,0,4,
      5,4,0,0,0,0,0,0,0,0,
      0,0,0,4,4,4,4,0,0,0,
      0,0,0,0,0,0,0,0,0,0]
def Extract(img_name, mask_name,o):
	# load image
	img = Image.open(img_name)
	# transfer to lab
	lab = rgb2lab(img)
	pixels = lab.load()

	# load alpha mattes
	mask = Image.open(mask_name) 
	m = mask.load()

	# get palettes from kmeans ( means = palettes)
	bins = {}
	for ii in range(img.width):
		for jj in range(img.height):
			if m[ii,jj] > 0:
				if bins.get(pixels[ii,jj]):
					bins[pixels[ii,jj]] += 1
				else:
					bins[pixels[ii,jj]] = 1
	bins = sample_bins(bins)
	k_palettes = num[o]
	means, _ = k_means(bins, k=k_palettes, init_mean=True)
	return means

def Recolor(img_name, mask_name, means,means2, o):
	# load image
	img = Image.open(img_name)
	# transfer to lab
	lab = rgb2lab(img)
	pixels = lab.load()
	# load alpha mattes
	mask = Image.open(mask_name) 
	m = mask.load()
	D = np.zeros((num[o],num[o]))
	
	for i in range(num[o]):
		for j in range(num[o]):
	            D[i,j] = (means[i,0]-means2[j,0])*(means[i,0]-means2[j,0])+(means[i,1]-means2[j,1])*(means[i,1]-means2[j,1])+(means[i,2]-means2[j,2])*(means[i,2]-means2[j,2])	

	NUM = np.zeros((num[o]))
	for i in range(num[o]):
		NUM[i] = i
	weight = 10000000
	INDEX = np.zeros((num[o]))        
	for item in permutations(NUM):
		d = 0
		for j in range(num[o]):
			d += D[j,int(item[j])]
		#print(d)
		if d<weight:
			weight = d
			INDEX = item
	INDEX2 = np.zeros((num[o])) 
	for i in range(num[o]):
		INDEX2[i] = int(INDEX[i])
	#print(INDEX2.astype(int))
	means2 = means2[INDEX2.astype(int)]
	# modified_p = [random.sample(range(255),3) for _ in range(k_palettes)]
	# sample grid from RGB colors and the get rbf weights
	sample_level = 16
	sample_colors = sample_RGB_color(sample_level)
	# used only when new image is loaded
	sample_weight_map = rbf_weights(means,sample_colors)

	# img color transfer given any modified_p 
	result = img_color_transfer(lab, means, means2, sample_weight_map, sample_colors, sample_level, m)
	result.save('result.png')
	return result
              


if __name__ == '__main__':
	img_name = 'multi/images/result34.png'
	mask_name = "multi/mattes/5_1.png"
	o =47
	means = Extract(img_name,mask_name,o)
	Recolor(img_name, mask_name, means, o)
