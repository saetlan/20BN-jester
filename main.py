import argparse
import configparser
from ast import literal_eval

import sys
import os
from math import ceil
import numpy as np

import lib.image as kmg
from lib.custom_callbacks import HistoryGraph
from lib.data_loader import DataLoader
from lib.utils import mkdirs
import lib.model as model
from lib.model_res import Resnet3DBuilder

from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD

def main(args):
    #Extracting the information from the configuration file
    mode         = config.get('general', 'mode')
    nb_frames    = config.getint('general', 'nb_frames')
    skip         = config.getint('general', 'skip')
    target_size  = literal_eval(config.get('general', 'target_size'))
    batch_size   = config.getint('general', 'batch_size')
    epochs       = config.getint('general', 'epochs')
    nb_classes   = config.getint('general', 'nb_classes')

    model_name   = config.get('path', 'model_name')
    data_root    = config.get('path', 'data_root')
    data_model   = config.get('path', 'data_model')
    data_vid     = config.get('path', 'data_vid')

    path_weights = config.get('path', 'path_weights')

    csv_labels   = config.get('path', 'csv_labels')
    csv_train    = config.get('path', 'csv_train')
    csv_val      = config.get('path', 'csv_val')
    csv_test     = config.get('path', 'csv_test')

    workers              = config.getint('option', 'workers')
    use_multiprocessing  = config.getboolean('option', 'use_multiprocessing')
    max_queue_size       = config.getint('option', 'max_queue_size')

    #Joining together the needed paths
    path_vid = os.path.join(data_root, data_vid)
    path_model = os.path.join(data_root, data_model, model_name)
    path_labels = os.path.join(data_root, csv_labels)
    path_train = os.path.join(data_root, csv_train)
    path_val = os.path.join(data_root, csv_val)
    path_test = os.path.join(data_root, csv_test)

    #Input shape of the input Tensor
    #inp_shape = (None, None, None, 3)
    inp_shape   = (nb_frames,) + target_size + (3,)

    
    if mode == 'train':
        data = DataLoader(path_vid, path_labels, path_train, path_val)

        #Creating the model and graph folder
        mkdirs(path_model, 0o755)
        mkdirs(os.path.join(path_model, "graphs"), 0o755)

        #Creating the generators for the training and validation set
        gen = kmg.ImageDataGenerator()
        gen_train = gen.flow_video_from_dataframe(data.train_df, path_vid, path_classes=path_labels, x_col='video_id', y_col="label", target_size=target_size, batch_size=batch_size, nb_frames=nb_frames, skip=skip, has_ext=True)
        gen_val = gen.flow_video_from_dataframe(data.val_df, path_vid, path_classes=path_labels, x_col='video_id', y_col="label", target_size=target_size, batch_size=batch_size, nb_frames=nb_frames, skip=skip, has_ext=True)
        
        """
        #Building model
        net = model.CNN3D(inp_shape=inp_shape,nb_classes=nb_classes, drop_rate=0.5)
        #Compiling model 
        net.compile(optimizer="Adadelta",
                   loss="categorical_crossentropy",
                   metrics=["accuracy", "top_k_categorical_accuracy"]) 
        """
        
        net = Resnet3DBuilder.build_resnet_101(inp_shape, nb_classes, drop_rate=0.5)
        opti = SGD(lr=0.01, momentum=0.9, decay= 0.0001, nesterov=False)
        net.compile(optimizer=opti,
                    loss="categorical_crossentropy",
                    metrics=["accuracy"]) 

        if(path_weights != "None"):
            print("Loading weights from : " + path_weights)
            net.load_weights(path_weights)


        #model_file_format_last = os.path.join(path_model,'model.{epoch:03d}.hdf5') 
        model_file_format_best = os.path.join(path_model,'model.best.hdf5') 

        checkpointer_best = ModelCheckpoint(model_file_format_best, monitor='val_acc',verbose=1, save_best_only=True, mode='max')
        #checkpointer_last = ModelCheckpoint(model_file_format_last, period=1)

        history_graph = HistoryGraph(model_path_name=os.path.join(path_model, "graphs"))

        #Get the number of sample in the training and validation set
        nb_sample_train = data.train_df["video_id"].size
        nb_sample_val   = data.val_df["video_id"].size

        #Launch the training 
        net.fit_generator(
                        generator=gen_train,
                        steps_per_epoch=ceil(nb_sample_train/batch_size),
                        epochs=epochs,
                        validation_data=gen_val,
                        validation_steps=ceil(nb_sample_val/batch_size),
                        shuffle=True,
                        verbose=1,
                        workers=workers,
                        max_queue_size=max_queue_size,
                        use_multiprocessing=use_multiprocessing,
                        callbacks=[checkpointer_best, history_graph],
        )
    elif mode == 'test':
        data = DataLoader(path_vid, path_labels, path_test=path_test)

        gen = kmg.ImageDataGenerator()
        gen_test = gen.flow_video_from_dataframe(data.test_df, path_vid, shuffle=False, path_classes=path_labels, class_mode=None, x_col='video_id', target_size=target_size, batch_size=batch_size, nb_frames=nb_frames, skip=skip, has_ext=True)

        #Building model
        net = Resnet3DBuilder.build_resnet_101(inp_shape, nb_classes)      

        if(path_weights != "None"):
            print("Loading weights from : " + path_weights)
            net.load_weights(path_weights)
        else: 
            sys.exit("<Error>: Specify a value for path_weights different from None when using test mode")

        #Get the number of sample in the test set 
        nb_sample_test = data.test_df["video_id"].size
 
        res = net.predict_generator(
                        generator=gen_test,
                        steps=ceil(nb_sample_test/batch_size),
                        verbose=1,
                        workers=workers,
                        use_multiprocessing=use_multiprocessing,
        )

        #Create an empty column called label
        data.test_df['label']=""

        #For each result get the string label and set it in the DataFrame
        for i, item in enumerate(res):
            item[item == np.max(item)]=1
            item[item != np.max(item)]=0
            label=data.categorical_to_label(item)

            #data.test_df.iloc[i,data.test_df.columns.get_loc('label')] = label
            data.test_df.at[i,'label'] = label #Faster than iloc

        #Save the resulting DataFrame to a csv
        data.test_df.to_csv(os.path.join(path_model,"prediction.csv"), sep=';', header=False, index=False)
    else:
        sys.exit("<Error>: Use either {train,test} mode")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config", help="Configuration file used to run the script", required=True)
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    main(config)
