import tensorflow as tf

class EquivariantLayer(tf.keras.layers.Layer):
    
    def __init__(self, output_channels, width=140, variation='sum', activation='relu', lambda_zero=False, gamma_zero=False, **kwargs):
        
        self.output_channels = output_channels
        self.width = width
        self.variation = variation
        self.activation = tf.keras.activations.get(activation)
        self.lambda_zero = lambda_zero
        self.gamma_zero = gamma_zero
        
        super(EquivariantLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.lam = self.add_weight(
            name='lambda',
            shape=((input_shape[0] if self.variation=='irc' else input_shape)[-1], self.output_channels),
            initializer = 'zeros' if self.lambda_zero else 'he_uniform',
            trainable = not self.lambda_zero
        )
        self.gam = self.add_weight(
            name='gamma',
            shape=((input_shape[0] if self.variation=='irc' else input_shape)[-1], self.output_channels),
            initializer = 'zeros' if self.gamma_zero else 'he_uniform',
            trainable = not self.gamma_zero
        )
        super(EquivariantLayer, self).build(input_shape)
     
    @tf.function
    def call(self,x):
        L = tf.einsum('ijk,kl->ijl', x[0] if self.variation=='irc' else x, self.lam)
        minipool = (
                 (lambda y: tf.math.reduce_sum(y, axis=-2, keepdims=True)) if self.variation=='sum' 
            else (lambda y: tf.math.reduce_max(y, axis=-2, keepdims=True)) if self.variation=='max'
            else (lambda y: tf.expand_dims(tf.einsum('ij,ijk->ik',y[1],y[0]),-2)) if self.variation=='irc'
            else None
        )
        R = tf.einsum('jk,ikl,lm->ijm', tf.expand_dims(tf.ones(self.width),1), minipool(x), self.gam)
        return self.activation(L + R)
    
    def compute_output_shape(self, input_shape):
        return (None, input_shape[-2], self.output_channels)