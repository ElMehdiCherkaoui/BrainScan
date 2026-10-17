import tensorflow as tf
from tensorflow.keras import layers, models
import time 
import tensorflow as tf





def build_brainscan_cnn():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu',input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4,activation='softmax')
        
    ])
    return model

def compile_model(model):
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    print("Model compiled successfully.")
    print(model.summary())
    return model

def adjust_hyperparameters(X_train, y_train, X_test, y_test, learning_rate=0.001, EPOCHS=20 , batch_size=32 , callbacks=None):
    model = build_brainscan_cnn()
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer, 
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    print("Model compiled successfully.")
    
    print("\nStarting training...")
    start_time = time.time()
    
    history = model.fit(
        x=X_train, 
        y=y_train,
        epochs=EPOCHS, 
        callbacks=callbacks,
        batch_size=batch_size,
        validation_data=(X_test, y_test)
    )
    
    end_time = time.time()
    total_seconds = end_time - start_time
    print(f"\nTraining completed in {total_seconds:.2f} seconds.")
    
    return model, history




