from keras.callbacks import Callback

import matplotlib as mpl
#mpl.use('TkAgg')  
import matplotlib.pyplot as plt

import json


class HistoryGraph(Callback):
    """Callback that records events into a `History` object.
    This callback is automatically applied to
    every Keras model. The `History` object
    gets returned by the `fit` method of models.
    """
    def __init__(self, model_path_name):
        self.model_path_name = model_path_name

    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        
        self.save_training_history(self.model_path_name, self.history)

    def save_training_history(self, path, history):
        """Save the history of the model and create graph for each metric and loss
        # Arguments
            path    : path where to save the graphs and history
            history : history of the model
        """
        for metric in history:
            if "val" not in metric:
                plt.clf()
                # list all data in history

                # summarize history for loss
                history[metric]=list(map(float, history[metric]))
                plt.plot(history[metric])
                plt.plot(history["val_" + metric])
                plt.title('model ' + metric)
                plt.ylabel(metric)
                plt.xlabel('epoch')
                plt.legend(['train', 'test'], loc='upper left')
                plt.gcf().savefig(path + '/'+metric+'_history' + '.jpg')

        # history to json file

        with open(path + '/log' + '.json', 'w') as fp:
            json.dump(history, fp, indent=True)