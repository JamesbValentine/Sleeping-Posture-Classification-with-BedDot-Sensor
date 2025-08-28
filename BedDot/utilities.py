import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils.input_data import read_data_sets
import utils.datasets as ds
import utils.augmentation as aug
import utils.helper as hlp
import random
import keras
from keras import backend as K
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn import metrics



def load_data(): #each day set the first 70% in time order as training set
    dataset = pd.read_csv('Sleep_toy.csv', skiprows=[0])
    dataset = dataset.reset_index(drop = True)
    total_rows = dataset.shape[0]
    dataset['date'] = dataset['Time'].str.slice(stop=10)
    dataset['Day'] = pd.Categorical(dataset.date).codes

    train = pd.DataFrame()
    test = pd.DataFrame()

    for i in range(15):
        Day = dataset[(dataset.Day == i)]
        total_rows = Day.shape[0]
        train_nrows = int(total_rows * 0.7)
        train_Day = Day.iloc[:train_nrows, :]
        test_Day = Day.iloc[train_nrows:, :]
        train = pd.concat([train, train_Day], axis=0)
        test = pd.concat([test, test_Day], axis=0)

    rows = train.values.tolist()
    random.shuffle(rows)
    train = pd.DataFrame(data=rows, columns=train.columns)

    rows = test.values.tolist()
    random.shuffle(rows)
    test = pd.DataFrame(data=rows, columns=test.columns)

    train_set_x_orig = train.iloc[:, 2:1000] # train set features
    train_set_y_orig = train.iloc[:, 1] # train set labels

    test_set_x_orig = test.iloc[:, 2:1000]# test set features
    test_set_y_orig = test.iloc[:, 1] # test set labels

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, dataset






# def load_data():
#     dataset = pd.read_csv('Sleep_toy.csv', skiprows=[0])
#     dataset = dataset.reset_index(drop = True)
#     total_rows = dataset.shape[0]
#     train_nrows = int(total_rows * 0.7)

#     train_set_x_orig = dataset.iloc[:train_nrows, -998:] # train set features
#     train_set_y_orig = dataset.iloc[:train_nrows, 1] # train set labels

#     test_set_x_orig = dataset.iloc[train_nrows:, -998:]# test set features
#     test_set_y_orig = dataset.iloc[train_nrows:, 1] # test set labels
    
#     # train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
#     # test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
#     return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, dataset



def show_dimensions(train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, dataset):
    shape_dict = {'Dataset': [], 'Shape': []}
    shape_dict['Dataset'] = ['train_set_x_orig', 'train_set_y_orig', 'test_set_x_orig', 'test_set_y_orig', 'whole_set']
    shape_dict['Shape'] = [train_set_x_orig.shape, train_set_y_orig.shape, test_set_x_orig.shape, test_set_y_orig.shape, dataset.shape]
    
    shape_table = pd.DataFrame.from_dict(shape_dict).set_index('Dataset')

    return shape_table



def transform_label(df):
    mapping = {'go_to_the_bed': 0, 'sleep_on_stomach': 1, 'sleep_on_left_side': 2, 'sleep_on_right_side': 3}
    new_array = np.array(df.replace(mapping))
    return new_array.astype('int64')


def mode_showing(n):
    train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, whole_set = load_data()
    fig, axs = plt.subplots(4, n, figsize=(20, 10))
    x = np.linspace(0, 10, 998)

    subsets = [whole_set[whole_set['Action'] == 'go_to_the_bed'],
               whole_set[whole_set['Action'] == 'sleep_on_stomach'],
               whole_set[whole_set['Action'] == 'sleep_on_left_side'],
               whole_set[whole_set['Action'] == 'sleep_on_right_side']]

    for i in range(4):
        for j in range(n):
            y = subsets[i].iloc[j, 2:1000]
            axs[i, j].plot(x, y)
            axs[i, j].text(0.5, 1.0, subsets[i].iloc[j,1], fontsize=10, color='black')
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])

    plt.show()



def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y



def show_data_agumentation_plots(train_x, Y, idx):
    train_x = np.array(train_x)
    x = train_x.reshape(train_x.shape[0], train_x.shape[1], 1)
    fig, axs = plt.subplots(5, 2, figsize = (20, 10))
    steps = np.arange(x.shape[1])

    axs[0,0].plot(steps, x[idx])  #original time series
    axs[0,0].set_title(f"Time Series Plot for sample: {idx+1}")
    axs[0,0].set_xticks([])

    axs[0,1].plot(steps, x[idx])
    axs[0,1].plot(steps, aug.jitter(x)[idx]) #Adding jittering, or noise, to the time series
    axs[0,1].set_title(f"Jitter Plot for sample: {idx+1}")
    axs[0,1].set_xticks([])

    axs[1,0].plot(steps, x[idx])
    axs[1,0].plot(steps, aug.scaling(x)[idx]) #Scaling each time series by a constant amount.
    axs[1,0].set_title(f"Scaling Plot for sample: {idx+1}")
    axs[1,0].set_xticks([])

    axs[1,1].plot(steps, x[idx])
    axs[1,1].plot(steps, aug.permutation(x)[idx]) #Random permutation of segments. 
    axs[1,1].set_title(f"Permutation Plot for sample: {idx+1}")
    axs[1,1].set_xticks([])

    axs[2,0].plot(steps, x[idx])
    axs[2,0].plot(steps, aug.magnitude_warp(x)[idx]) #The magnitude of each time series is multiplied by a curve created by cubicspline with a set number of knots at random magnitudes.
    axs[2,0].set_title(f"Magnitude Warp Plot for sample: {idx+1}")
    axs[2,0].set_xticks([])

    axs[2,1].plot(steps, x[idx])
    axs[2,1].plot(steps, aug.time_warp(x)[idx]) #Random smooth time warping.
    axs[2,1].set_title(f"Time Warp Plot for sample: {idx+1}")
    axs[2,1].set_xticks([])

    axs[3,0].plot(steps, x[idx])
    axs[3,0].plot(steps, aug.rotation(x)[idx]) #For 1D time series, randomly flipping.
    axs[3,0].set_title(f"Rotation Plot for sample: {idx+1}")
    axs[3,0].set_xticks([])

    axs[3,1].plot(steps, x[idx])
    axs[3,1].plot(steps, aug.window_slice(x)[idx]) #Cropping the time series by the reduce_ratio
    axs[3,1].set_title(f"Window Slice Plot for sample: {idx+1}")
    axs[3,1].set_xticks([])

    axs[4,0].plot(steps, x[idx])
    axs[4,0].plot(steps, aug.window_warp(x)[idx]) #Randomly warps a window by scales.
    axs[4,0].set_title(f"Window Warp Plot for sample: {idx+1}")
    axs[4,0].set_xticks([])

    axs[4,1].plot(steps, x[idx])
    axs[4,1].plot(steps, aug.spawner(x, Y)[idx])
    axs[4,1].set_title(f"Spawner Plot for sample: {idx+1}")
    axs[4,1].set_xticks([])

    # axs[2,2].plot(steps, x[idx])
    # axs[2,2].plot(steps, aug.wdba(x, Y)[idx])

    # axs[2,3].plot(steps, x[idx])
    # axs[2,3].plot(steps, aug.discriminative_guided_warp(x, Y)[idx])

    plt.show()


def augment_training_set(train_x, train_set_y): #train_x is pd.DataFrame, train_set_y is np.array
    #train_x should be a training set for features finished standardlization.
    #train_set_y should be a training set of labels after changing strings to np.float64.
    train_x = np.array(train_x)
    x = train_x.reshape(train_x.shape[0], train_x.shape[1], 1)
    combined_arr_jittering = np.column_stack((np.squeeze(aug.jitter(x)), train_set_y))
    combined_df_jittering = pd.DataFrame(combined_arr_jittering)
    t1 = pd.DataFrame(np.column_stack((train_x, train_set_y)))
    combined_df_jittering = pd.concat([t1, combined_df_jittering], axis=0)

    combined_df_scaling = pd.DataFrame(np.column_stack((np.squeeze(aug.scaling(x)), train_set_y)))
    combined_df_scaling = pd.concat([combined_df_jittering, combined_df_scaling], axis=0)

    combined_df_permutation = pd.DataFrame(np.column_stack((np.squeeze(aug.permutation(x)), train_set_y)))
    combined_df_permutation = pd.concat([combined_df_scaling, combined_df_permutation], axis=0)

    combined_df_magnitude_warp = pd.DataFrame(np.column_stack((np.squeeze(aug.magnitude_warp(x)), train_set_y)))
    combined_df_magnitude_warp = pd.concat([combined_df_permutation, combined_df_magnitude_warp], axis=0)

    combined_df_time_warp = pd.DataFrame(np.column_stack((np.squeeze(aug.time_warp(x)), train_set_y)))
    combined_df_time_warp = pd.concat([combined_df_magnitude_warp, combined_df_time_warp], axis=0)

    combined_df_rotation = pd.DataFrame(np.column_stack((np.squeeze(aug.rotation(x)), train_set_y)))
    combined_df_rotation = pd.concat([combined_df_time_warp, combined_df_rotation], axis=0)

    combined_df_window_slice = pd.DataFrame(np.column_stack((np.squeeze(aug.window_slice(x)), train_set_y)))
    combined_df_window_slice = pd.concat([combined_df_rotation, combined_df_window_slice], axis=0)

    combined_df_window_warp = pd.DataFrame(np.column_stack((np.squeeze(aug.window_warp(x)), train_set_y)))
    augmentation_set = pd.concat([combined_df_window_slice, combined_df_window_warp], axis=0)

    rows = augmentation_set.values.tolist()
    random.shuffle(rows)
    augmentation_set = pd.DataFrame(data=rows, columns=augmentation_set.columns)

    return augmentation_set


def load_data_for_one_day(day):
    dataset = pd.read_csv('Sleep_toy.csv', skiprows=[0])
    dataset = dataset.reset_index(drop = True)
    total_rows = dataset.shape[0]
    dataset['date'] = dataset['Time'].str.slice(stop=10)
    dataset['Day'] = pd.Categorical(dataset.date).codes

    test = dataset[(dataset.Day == day-1)]
    train = dataset[~(dataset.Day == day-1)]

    rows = train.values.tolist()
    random.shuffle(rows)
    train_shuffled = pd.DataFrame(data=rows, columns=train.columns)

    rows = test.values.tolist()
    random.shuffle(rows)
    test_shuffled = pd.DataFrame(data=rows, columns=test.columns)

    return train_shuffled, test_shuffled

def Step4_create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(998),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(1000, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1000, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1000, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(20, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(4, activation="softmax")
    ])
    model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['acc',f1_m,precision_m,recall_m])
    return model



def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def auc(y_true, y_pred):
    return metrics.roc_auc_score(K.eval(y_true), K.eval(y_pred))



def NN_augmentation_fit_15_days():
    dict = {}
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='loss',
        patience=5,
        ),
        keras.callbacks.TensorBoard(
            log_dir='my_log_dir',
            histogram_freq=1,
            embeddings_freq=1,
        )
    ]

    for i in range(15):
        key = f"day_{i}"
        train, test = load_data_for_one_day(i+1)
        train_x = train.iloc[:, 2:1000]
        train_y = train.iloc[:, 1]
        train_set_y = transform_label(train_y)
        augmentation_train_set = augment_training_set(train_x, train_set_y)
        augmentation_train_x = augmentation_train_set.iloc[:, :998]
        augmentation_train_y = augmentation_train_set.iloc[:, 998]
        from sklearn.preprocessing import StandardScaler
        # create a StandardScaler object
        scaler = StandardScaler()
        # fit the scaler to your dataframe and transform it
        augmentation_train_x_std = scaler.fit_transform(augmentation_train_x)
        augmentation_train_x = pd.DataFrame(augmentation_train_x_std)
        augmentation_Y = pd.get_dummies(augmentation_train_y)
        augmentation_Y = augmentation_Y.replace({True: 1, False: 0}).astype(float)

        model = Step4_create_model()

        history = model.fit(
            augmentation_train_x.values,
            augmentation_Y.values,
            validation_split=0.3,
            epochs=100,
            #callbacks=callbacks,
            batch_size=100,
            callbacks=[callbacks],
        )
        dict[key] = history

        model.save("my_model_day_"+str(i+1)+".h5")

        print("-------------------------------------------")
        print('Day '+str(i+1)+' Finished!')
        print("-------------------------------------------")

        # result = model1.predict(test_x)
        # t = np.argmax(result, axis=1) 
        # accuracy_score(test_set_y,t)

    return dict



def NN_augmentation_get_15_days_prediction():
    data_list = []
    for i in range(15):
        key = f"day_{i}"
        train, test = load_data_for_one_day(i+1)
        test_x = test.iloc[:, 2:1000]
        test_y = test.iloc[:, 1]
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        # fit the scaler to your dataframe and transform it
        test_x_std = scaler.fit_transform(test_x)
        test_x = pd.DataFrame(test_x_std)
        new_values = pd.Series(['go_to_the_bed','sleep_on_stomach','sleep_on_left_side','sleep_on_right_side'])
        test_y = pd.concat([new_values, test_y]).reset_index(drop=True)
        test_y = transform_label(test_y)
        test_y = pd.get_dummies(test_y)
        test_y = test_y.replace({True: 1, False: 0}).astype(float)
        test_y = test_y.iloc[4:]

        model = keras.models.load_model("my_model_day_"+str(i+1)+".h5", custom_objects={'f1_m': f1_m, 'precision_m': precision_m, 'recall_m': recall_m})
        # model = keras.models.load_model("my_model.h5")

        loss, accuracy, f1_score, precision, recall = model.evaluate(test_x, test_y, verbose=0)
        # result = np.array(loss, accuracy, f1_score, precision, recall)

        df = pd.DataFrame({"loss": [loss], "accuracy": [accuracy], "f1_score": [f1_score], "precision": [precision], "recall": [recall]}, index=["Day"+str(i+1)])
        data_list.append(df)

        # result = model.predict(test_x)
        # result_df[key] = result 
    
    total = pd.DataFrame({"loss": [loss], "accuracy": [accuracy], "f1_score": [f1_score], "precision": [precision], "recall": [recall]}, index=["Overall"])
    result_df = pd.concat(data_list)

    return result_df

def load_larger_dataset():
    dataset = pd.read_csv('signal_1000_posture_4.csv')
    dataset = dataset.reset_index(drop = True)
    total_rows = dataset.shape[0]
    dataset['date'] = dataset['Time'].str.slice(stop=10)
    dataset['Day'] = pd.Categorical(dataset.date).codes

    train = pd.DataFrame()
    test = pd.DataFrame()

    for i in range(15):
        Day = dataset[(dataset.Day == i)]
        total_rows = Day.shape[0]
        train_nrows = int(total_rows * 0.7)
        train_Day = Day.iloc[:train_nrows, :]
        test_Day = Day.iloc[train_nrows:, :]
        train = pd.concat([train, train_Day], axis=0)
        test = pd.concat([test, test_Day], axis=0)

    Day = dataset[(dataset['Day'] >= 15) & (dataset['Day'] <= 43)]
    test = pd.concat([test, Day], axis=0)

    rows = train.values.tolist()
    random.shuffle(rows)
    train = pd.DataFrame(data=rows, columns=train.columns)

    rows = test.values.tolist()
    random.shuffle(rows)
    test = pd.DataFrame(data=rows, columns=test.columns)

    train_set_x_orig = train.iloc[:, 2:1000] # train set features
    train_set_y_orig = train.iloc[:, 1] # train set labels

    test_set_x_orig = test.iloc[:, 2:1000]# test set features
    test_set_y_orig = test.iloc[:, 1] # test set labels

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, dataset

def load_larger_dataset_first_scenario():
    dataset = pd.read_csv('signal_1000_posture_4.csv')
    dataset = dataset.reset_index(drop = True)
    total_rows = dataset.shape[0]
    train_nrows = int(total_rows * 0.7)

    train = dataset.iloc[:train_nrows, :]
    test = dataset.iloc[train_nrows: , :]

    rows = train.values.tolist()
    random.shuffle(rows)
    train = pd.DataFrame(data=rows, columns=train.columns)

    rows = test.values.tolist()
    random.shuffle(rows)
    test = pd.DataFrame(data=rows, columns=test.columns)

    train_set_x_orig = train.iloc[:, 2:] # train set features
    train_set_y_orig = train.iloc[:, 1] # train set labels

    test_set_x_orig = test.iloc[:, 2:]# test set features
    test_set_y_orig = test.iloc[:, 1] # test set labels


    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, dataset, train, test



def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(1000), # dimension of X matrix
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(10000, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(5000, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1000, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(200, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(20, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(4, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['acc'])
    return model




def NN_augmentation_fit_44_days():
    dict = {}
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='loss',
        patience=5,
        ),
        keras.callbacks.TensorBoard(
            log_dir='my_log_dir',
            histogram_freq=1,
            embeddings_freq=1,
        )
    ]

    for i in range(44):
        key = f"day_{i}"
        train, test = load_data_for_one_day(i+1)
        train_x = train.iloc[:, 2:1002]
        train_y = train.iloc[:, 1]
        train_set_y = transform_label(train_y)
        # augmentation_train_set = augment_training_set(train_x, train_set_y)
        # augmentation_train_x = augmentation_train_set.iloc[:, :998]
        # augmentation_train_y = augmentation_train_set.iloc[:, 998]
        from sklearn.preprocessing import StandardScaler
        # create a StandardScaler object
        scaler = StandardScaler()
        # fit the scaler to your dataframe and transform it
        train_x_std = scaler.fit_transform(train_x)
        train_x = pd.DataFrame(train_x_std)
        Y = pd.get_dummies(train_set_y)
        Y = Y.replace({True: 1, False: 0}).astype(float)

        model = create_model()

        history = model.fit(
            train_x.values,
            Y.values,
            validation_split=0.3,
            epochs=100,
            #callbacks=callbacks,
            batch_size=100,
            callbacks=[callbacks],
        )
        dict[key] = history

        model.save("429_my_model_day_"+str(i+1)+".h5")

        print("-------------------------------------------")
        print('Day '+str(i+1)+' Finished!')
        print("-------------------------------------------")

        # result = model1.predict(test_x)
        # t = np.argmax(result, axis=1) 
        # accuracy_score(test_set_y,t)

    return dict


def NN_augmentation_get_44_days_prediction():
    data_list = []
    for i in range(44):
        key = f"day_{i}"
        train, test = load_data_for_one_day(i+1)
        test_x = test.iloc[:, 2:1002]
        test_y = test.iloc[:, 1]
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        # fit the scaler to your dataframe and transform it
        test_x_std = scaler.fit_transform(test_x)
        test_x = pd.DataFrame(test_x_std)
        new_values = pd.Series(['go_to_the_bed','sleep_on_stomach','sleep_on_left_side','sleep_on_right_side'])
        test_y = pd.concat([new_values, test_y]).reset_index(drop=True)
        test_y = transform_label(test_y)
        test_y = pd.get_dummies(test_y)
        test_y = test_y.replace({True: 1, False: 0}).astype(float)
        test_y = test_y.iloc[4:]

        model = keras.models.load_model("429_my_model_day_"+str(i+1)+".h5", custom_objects={'f1_m': f1_m, 'precision_m': precision_m, 'recall_m': recall_m})
        # model = keras.models.load_model("my_model.h5")

        loss, accuracy, f1_score, precision, recall = model.evaluate(test_x, test_y, verbose=0)
        # result = np.array(loss, accuracy, f1_score, precision, recall)

        df = pd.DataFrame({"loss": [loss], "accuracy": [accuracy], "f1_score": [f1_score], "precision": [precision], "recall": [recall]}, index=["Day"+str(i+1)])
        data_list.append(df)

        # result = model.predict(test_x)
        # result_df[key] = result 
    
    total = pd.DataFrame({"loss": [loss], "accuracy": [accuracy], "f1_score": [f1_score], "precision": [precision], "recall": [recall]}, index=["Overall"])
    result_df = pd.concat(data_list)

    return result_df


