import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
import cv2

class EnhancedAlphabetTrainer:
    def __init__(self, data_dir="data/alphabet_images", model_dir="models"):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.img_size = (224, 224)
        self.batch_size = 16
        
    def create_data_generators(self):
        """Create augmented data generators"""
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=0.2
        )
        
        train_generator = train_datagen.flow_from_directory(
            self.data_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            color_mode='rgb'
        )
        
        val_generator = train_datagen.flow_from_directory(
            self.data_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            color_mode='rgb'
        )
        
        return train_generator, val_generator
    
    def build_model(self, num_classes):
        """Build EfficientNet-based model"""
        base_model = EfficientNetB0(
            weights=None,  # Don't use pre-trained weights to avoid shape mismatch
            include_top=False,
            input_shape=(*self.img_size, 3)
        )
        
        # Freeze base model layers initially
        base_model.trainable = False
        
        # Add custom top layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.3)(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = BatchNormalization()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.2)(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        return model, base_model
    
    def train(self):
        """Train the enhanced alphabet model"""
        print("🚀 Training Enhanced Alphabet Model with Transfer Learning...")
        
        # Create data generators
        train_gen, val_gen = self.create_data_generators()
        
        # Build model
        model, base_model = self.build_model(train_gen.num_classes)
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=5, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=3),
            ModelCheckpoint(
                os.path.join(self.model_dir, 'best_alphabet_model.keras'),
                save_best_only=True,
                monitor='val_accuracy'
            )
        ]
        
        # Initial training with frozen base
        print("📊 Phase 1: Training with frozen base...")
        history1 = model.fit(
            train_gen,
            epochs=10,
            validation_data=val_gen,
            callbacks=callbacks
        )
        
        # Fine-tuning
        print("🔧 Phase 2: Fine-tuning...")
        base_model.trainable = True
        
        # Use lower learning rate for fine-tuning
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        history2 = model.fit(
            train_gen,
            epochs=20,
            validation_data=val_gen,
            callbacks=callbacks
        )
        
        # Save final model
        model.save(os.path.join(self.model_dir, 'enhanced_alphabet_model.keras'))
        
        # Save label mapping
        with open(os.path.join(self.model_dir, 'enhanced_alphabet_labels.json'), 'w') as f:
            json.dump(train_gen.class_indices, f, indent=2)
        
        print("✅ Enhanced alphabet training complete!")
        return model

if __name__ == "__main__":
    trainer = EnhancedAlphabetTrainer()
    trainer.train()
