import os
import tensorflow as tf
import cv2
import numpy as np

model_path = os.path.join(os.getcwd(), 'model/model.h5')
print('###############################################')
print('Loading model - {}'.format(model_path))
print('###############################################')

model= tf.keras.models.load_model(model_path)
print('###############################################')
print('Loading complete')
print('###############################################')
print('\n\n\n\n\n\n\n')

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

'''
WRITE A FUNCTION THAT ACCEPTS AN IMAGE AND RETURN PREDICTED LABEL.
'''
def get_image_name(img):
	img2 = img.reshape((1, 28, 28))
	predictions = model.predict(img2)
	predict_arg = np.argmax(predictions[0])
	predicted_val = class_names[predict_arg]
	return predicted_val

img = cv2.imread('/home/gaurav.gupta/projects/ai_ml_training/images/test_file.jpg', 0)
print('Input image looks like \n\n- {}\n\n\n'.format(img))
print('Predicted value is  - {}'.format(get_image_name(img)))
