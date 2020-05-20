import matplotlib.pyplot as plt
    
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