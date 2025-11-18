"""
Simple script to test if the saved models can be loaded successfully.
"""
import os
import tensorflow as tf

# Custom attention layer from the notebook
class ChannelAttention(layers.Layer):
    """Squeeze-and-Excitation block for channel attention"""
    def __init__(self, ratio=8, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.ratio = ratio
        
    def build(self, input_shape):
        channels = input_shape[-1]
        self.shared_dense_one = layers.Dense(channels // self.ratio, activation='relu')
        self.shared_dense_two = layers.Dense(channels, activation='sigmoid')
        
    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=[1, 2], keepdims=True)
        
        avg_out = self.shared_dense_two(self.shared_dense_one(avg_pool))
        max_out = self.shared_dense_two(self.shared_dense_one(max_pool))
        
        attention = avg_out + max_out
        return inputs * attention

print("="*70)
print("MODEL VERIFICATION TEST")
print("="*70)

# Test each model
models = [
    'best_VGG16_Careful.keras',
    'best_CustomCNN_Attention.keras',
    'best_CustomCNN_Wide.keras',
    'Final_Ensemble_VGG_Custom.keras'
]

for model_name in models:
    if os.path.exists(model_name):
        print(f"\n✓ Testing {model_name}...")
        try:
            if 'Attention' in model_name:
                # Load with custom objects for attention model
                model = tf.keras.models.load_model(
                    model_name,
                    custom_objects={'ChannelAttention': ChannelAttention}
                )
            else:
                model = tf.keras.models.load_model(model_name)
            
            print(f"  ✓ Model loaded successfully!")
            print(f"  - Input shape: {model.input_shape}")
            print(f"  - Output shape: {model.output_shape}")
            print(f"  - Total parameters: {model.count_params():,}")
        except Exception as e:
            print(f"  ✗ Error loading model: {e}")
    else:
        print(f"\n✗ Model file not found: {model_name}")

print("\n" + "="*70)
print("MODEL VERIFICATION COMPLETE")
print("="*70)
