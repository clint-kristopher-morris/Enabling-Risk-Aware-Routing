import os
import pickle


def save_obj(obj, file_name):
    """pickle object
    """
    if not os.path.exists('obj/'):
        os.makedirs('obj/')
    with open('obj/' + file_name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(file_name):
    """load pickled object
    """
    with open('obj/' + file_name + '.pkl', 'rb') as f:
        return pickle.load(f)
