import wandb
import matplotlib.pyplot as plt

from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

from sklearn.model_selection import train_test_split

from src.utils import WandbClassificationCallback

def main():
    wandb.init(project="ufpa-mnist")
    config = wandb.config
    config.epochs = 20
    config.batch_size = 32

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)    

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    model = Sequential()
    model.add(Conv2D(16, kernel_size=3, activation='relu', input_shape=(28,28,1)))
    model.add(MaxPooling2D())
    model.add(Conv2D(8, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print(model.summary())

    wandb_callback = WandbClassificationCallback(
        log_confusion_matrix=True, 
        validation_data=(X_test, y_test),
        labels=list(range(10))
    )

    model.fit(X_train, y_train, 
        batch_size=config.batch_size,
        epochs=config.epochs, 
        verbose=1,
        validation_data=(X_test, y_test),
        callbacks=[wandb_callback]
    )

if __name__ == '__main__':
    main()