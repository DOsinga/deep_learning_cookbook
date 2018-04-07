from keras.models import load_model, model_from_config
import h5py
from keras import backend as K

K.set_learning_phase(0)
char_cnn = load_model('zoo/07.2 char_cnn_model.h5')
config = char_cnn.get_config()
weights = char_cnn.get_weights()

config = {'config': config,
          'class_name': 'Model'}
new_model = model_from_config(config)
new_model.set_weights(weights)