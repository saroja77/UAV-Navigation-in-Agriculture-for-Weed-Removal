import datetime 
start_sc = datetime.datetime.now()

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys
import argparse
import tensorflow as tf
import segmentation_models as sm
from tensorflow import keras


parser = argparse.ArgumentParser()
parser.add_argument('--input_folder', required=True, help="Please enter the absolute path of the folder with images.")
parser.add_argument('--output_folder', required=True, help="Please enter the absolute path where the extracted files will be saved.")
args = parser.parse_args()

def _is_valid_directory(arg):
    if not os.path.isdir(arg):
        parser.error('The directory {} does not exist!'.format(arg))
    else:
        return arg

images = args.input_folder
save_dir = _is_valid_directory(args.output_folder)

overlays_dir = os.path.join(save_dir, 'overlays')
masks_dir = os.path.join(save_dir, 'masks')

os.makedirs(overlays_dir, exist_ok=True)
os.makedirs(masks_dir, exist_ok=True)

sm.set_framework('tf.keras')

extensions = ["*.png", "*.jpg", ".jpeg"]

predict_imgs = []
info = []
for ext in extensions:
	for img_path in glob.glob(os.path.join(images, ext)):
		img_name = os.path.basename(img_path)
		img = cv2.imread(img_path, cv2.IMREAD_COLOR)
		height, width, channels = img.shape

		q_height, mod_height = divmod(height, 32)
		if mod_height != 0:
			new_height = (q_height + 1)*32
		else:
			new_height = height

		q_width, mod_width = divmod(width, 32)
		if mod_width != 0:
			new_width = (q_width + 1)*32
		else:
			new_width = width

		result = cv2.copyMakeBorder(img, 0, int(new_height-height), 0, int(new_width-width), cv2.BORDER_CONSTANT, None, value = 0)
		img_reshaped = cv2.cvtColor(result, cv2.COLOR_BGR2RGB) 

		info.append([os.path.basename(img_path), img.shape])
		predict_imgs.append(img_reshaped)
	   

predict_imgs_arr = np.array(predict_imgs)

classes = ['weeds', 'background']
activation = 'sigmoid' if len(classes) == 1 else 'softmax'

BACKBONE = 'efficientnetb1'
preprocess_input = sm.get_preprocessing(BACKBONE)

model = sm.Unet(BACKBONE, classes=len(classes), activation=activation)
model.load_weights('./weights0300.hdf5')

times = []
for pred_img, img_specs in zip(predict_imgs_arr, info):
	pred_img_input = np.expand_dims(pred_img, 0)
	test_img_input = preprocess_input(pred_img_input)

	start = datetime.datetime.now()
	mask_unet = model.predict(test_img_input)
	stop = datetime.datetime.now()
	times.append((stop-start).total_seconds())
	
	mask = np.argmax(mask_unet, axis=3)[0,:,:]*255	
	mask_3ch = np.stack((mask, )*3, axis = -1)
	mask_3ch = mask_3ch.astype('uint8')
	mask_3ch[mask==255] = [150, 10, 150]	

	result = cv2.addWeighted(pred_img, 1, mask_3ch, 0.9, 0.7, dtype = cv2.CV_8UC3)
	mask_reshaped = mask_3ch[:img_specs[1][0], :img_specs[1][1]]
	result_reshaped = result[:img_specs[1][0], :img_specs[1][1]]

	f = plt.figure()
	f.set_figheight(result_reshaped.shape[0] / f.get_dpi())
	f.set_figwidth(result_reshaped.shape[1] / f.get_dpi())
	ax = plt.Axes(f, [0., 0., 1., 1.])
	ax.set_axis_off()
	f.add_axes(ax)
	ax.imshow(result_reshaped)
	f.savefig('{}/{}'.format(overlays_dir, img_specs[0]))
	plt.close()

	f = plt.figure()
	f.set_figheight(mask_reshaped.shape[0] / f.get_dpi())
	f.set_figwidth(mask_reshaped.shape[1] / f.get_dpi())
	ax = plt.Axes(f, [0., 0., 1., 1.])
	ax.set_axis_off()
	f.add_axes(ax)
	ax.imshow(mask_reshaped)
	f.savefig('{}/{}'.format(masks_dir, img_specs[0]))
	plt.close()

try:
	print('Average prediction time for each image: {}s'.format(sum(times[1:])/len(times[1:])))
except:
	print('No need of calculating average time because the number of input images is one!')

stop_sc = datetime.datetime.now()
print('Execution time of this module: {}s'.format(stop_sc-start_sc))
print('Done!')
