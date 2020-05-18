import matplotlib.pyplot as plt

def print_accuracy_report(X_train, X_test, y_train, y_test, model):
     
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
    import numpy as np
    from keras.preprocessing.image import ImageDataGenerator
    
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_generator = test_datagen.flow_from_directory(
    test_folder,
    target_size=input_size,
    shuffle=False,
    batch_size=v_batch_size,
    class_mode='binary')
    
    y_true=test_generator.classes
    y_pred = model.predict_generator(test_generator)
    y_pred = np.rint(y_pred)

    print(classification_report(y_true, y_pred, labels=[0,1]))


def plot_con_matrix(y_test, y_pred, class_names, cmap=plt.cm.Blues):
    
    from keras.preprocessing.image import ImageDataGenerator
    import numpy as np
    import matplotlib.pyplot as plt
    import itertools
    from sklearn.metrics import confusion_matrix
    import os
    
    SEED = 1234
    batch_size = 64
    v_batch_size = 64
    input_size = (32, 32)
    input_shape = input_size + (3, )
    
    new_dir = 'split/'
    test_folder = os.path.join(new_dir, 'test')
    
    
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_generator = test_datagen.flow_from_directory(
    test_folder,
    target_size=input_size,
    shuffle=False,
    batch_size=v_batch_size,
    class_mode='binary')
    
    y_true=test_generator.classes
    y_pred = model.predict_generator(test_generator)
    y_pred = np.rint(y_pred)
    
    plt.grid(b=None)
    
    cnf_matrix = confusion_matrix(y_true, y_pred)
    
    plt.imshow(cnf_matrix, cmap=cmap) 

    # Add title and axis labels
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Add appropriate axis scales
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)


    # Add labels to each cell
    thresh = cnf_matrix.max() / 2. # Used for text coloring below
    
    # Here we iterate through the confusion matrix and append labels to our visualization 
    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, cnf_matrix[i, j],
                 horizontalalignment='center',
                 fontsize=16,
                 color='white' if cnf_matrix[i, j] > thresh else 'black')

    
    plt.colorbar()
    plt.show()
    
def visualize_training_results(results):
        history = results.history
        plt.figure()
        plt.plot(history['val_loss'])
        plt.plot(history['loss'])
        plt.legend(['val_loss', 'loss'])
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()
    
        plt.figure()
        plt.plot(history['val_acc'])
        plt.plot(history['acc'])
        plt.legend(['val_acc', 'acc'])
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.show()