# import tensorflow as tf

# class EquivariantLayer(tf.keras.layers.Layer):
    
#     def __init__(self, output_channels, width=140, variation='sum', activation='relu', lambda_zero=False, gamma_zero=False, **kwargs):
        
#         self.output_channels = output_channels
#         self.width = width
#         self.variation = variation
#         self.activation = tf.keras.activations.get(activation)
#         self.lambda_zero = lambda_zero
#         self.gamma_zero = gamma_zero
        
#         super(EquivariantLayer, self).__init__(**kwargs)
    
#     def build(self, input_shape):
#         self.lam = self.add_weight(
#             name='lambda',
#             shape=((input_shape[0] if self.variation=='irc' else input_shape)[-1], self.output_channels),
#             initializer = 'zeros' if self.lambda_zero else 'he_uniform',
#             trainable = not self.lambda_zero
#         )
#         self.gam = self.add_weight(
#             name='gamma',
#             shape=((input_shape[0] if self.variation=='irc' else input_shape)[-1], self.output_channels),
#             initializer = 'zeros' if self.gamma_zero else 'he_uniform',
#             trainable = not self.gamma_zero
#         )
#         super(EquivariantLayer, self).build(input_shape)
     
#     # @tf.function
#     def call(self,x):
#         L = tf.einsum('ijk,kl->ijl', x[0] if self.variation=='irc' else x, self.lam)
#         minipool = (
#                  (lambda y: tf.math.reduce_sum(y, axis=-2, keepdims=True)) if self.variation=='sum' 
#             else (lambda y: tf.math.reduce_max(y, axis=-2, keepdims=True)) if self.variation=='max'
#             else (lambda y: tf.expand_dims(tf.einsum('ij,ijk->ik',y[1],y[0]),-2)) if self.variation=='irc'
#             else None
#         )
#         R = tf.einsum('jk,ikl,lm->ijm', tf.expand_dims(tf.ones(self.width),1), minipool(x), self.gam)
#         return self.activation(L + R)
    
#     def compute_output_shape(self, input_shape):
#         return (None, input_shape[-2], self.output_channels)

import tensorflow as tf

class EquivariantLayer(tf.keras.layers.Layer):
    
    def __init__(self, output_channels, width=140, variation='sum', activation='relu', lambda_zero=False, gamma_zero=False, **kwargs):
    
        super(EquivariantLayer, self).__init__(**kwargs)
        tf.keras.backend.set_floatx('float64')
        
        self.output_channels = output_channels
        self.width = width
        self.variation = variation
        self.activation = tf.keras.activations.get(activation)
        self.lambda_zero = lambda_zero
        self.gamma_zero = gamma_zero
        
    def build(self, input_shape):
        self.lam = self.add_weight(
            name='lambda',
            dtype=tf.float64,
            shape=((input_shape[0] if self.variation=='irc' else input_shape)[-1], self.output_channels),
            initializer = 'zeros' if self.lambda_zero else 'he_uniform',
            trainable = not self.lambda_zero
        )
        self.gam = self.add_weight(
            name='gamma',
            dtype=tf.float64,
            shape=((input_shape[0] if self.variation=='irc' else input_shape)[-1], self.output_channels),
            initializer = 'zeros' if self.gamma_zero else 'he_uniform',
            trainable = not self.gamma_zero
        )
        super(EquivariantLayer, self).build(input_shape)
     
#     @tf.function
    def call(self,x):
        L = tf.einsum('ijk,kl->ijl', x[0] if self.variation=='irc' else x, self.lam)
        minipool = (
                 (lambda y: tf.math.reduce_sum(y, axis=-2, keepdims=True)) if self.variation=='sum' 
            else (lambda y: tf.math.reduce_max(y, axis=-2, keepdims=True)) if self.variation=='max'
            else (lambda y: tf.expand_dims(tf.einsum('ij,ijk->ik',y[1],y[0]),-2)) if self.variation=='irc'
            else None
        )
        R = tf.einsum('jk,ikl,lm->ijm', tf.expand_dims(tf.ones(self.width, dtype=tf.float64),1), minipool(x), self.gam)
        return self.activation(L + R)
    
    def compute_output_shape(self, input_shape):
        return (None, input_shape[-2], self.output_channels)
    

class EquivariantModel(tf.keras.Model):
    
    def __init__(self, model, ppm_sizes, equi_channels, equi_type, projection, f_sizes, lambda_zero, gamma_zero, equi_act='relu', dropout=0.0, **kwargs):
        
        super().__init__(kwargs)
        
        self.model = model
        self.ppm_sizes = ppm_sizes
        self.equi_channels = equi_channels
        self.equi_type = equi_type
        self.projection = projection
        self.f_sizes = f_sizes
        self.lambda_zero = lambda_zero
        self.gamma_zero = gamma_zero
        self.equi_act = equi_act
        self.dropout = dropout
        
        self.is_evefn = self.model == 'ev-efn'
        self.is_evpfn = self.model == 'ev-pfn'
        self.is_evpfnid = self.model == 'ev-pfn-id'
        
        tf.keras.backend.set_floatx('float64')
        
        if self.is_evpfnid:
            self.encoder = tf.keras.layers.Lambda(lambda x: tf.concat([x[0], tf.one_hot(tf.cast(x[1],'int64'),14, dtype=tf.float64)], axis=-1), name='particle-info')
        
        # ppm layers
        self.ppm_layers = [
            tf.keras.layers.TimeDistributed(
                    tf.keras.layers.Dense(s, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.HeUniform()),
                    name=f'ppm-{i+1}'
            ) for i, s in enumerate(self.ppm_sizes)
        ]
        
        # equivariant layers
        self.equi_layers = [
            EquivariantLayer(s,
                             width=139,
                             variation=self.equi_type,
                             activation=self.equi_act,
                             lambda_zero=self.lambda_zero,
                             gamma_zero=self.gamma_zero,
                             name=f'{self.equi_type}-equivariant-{i+1}'
            ) for i, s in enumerate(self.equi_channels)
        ]
            
        # projection
        if self.is_evefn:
            self.z_weighting  = tf.keras.layers.Lambda(lambda y: tf.expand_dims(tf.einsum('ij,ijk->ik',y[1],y[0]),-2), name='z-weighting')
        self.proj = (
                 tf.keras.layers.Lambda(lambda x: tf.reduce_max(x,axis=1), name='maxpooling') if self.projection=='max'
            else tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x,axis=1), name='summation') if self.projection=='sum'
            else None
        )
        
        # F layers
        self.F_layers = [
            tf.keras.layers.Dense(s,
                                  activation=tf.nn.relu, 
                                  kernel_initializer=tf.keras.initializers.HeUniform(), 
                                  name=f'F-{i+1}'
            ) for i, s in enumerate(self.f_sizes)
        ]
        self.dropout_layers = [
            tf.keras.layers.Dropout(self.dropout, name=f'dropout-{i+1}') for i, s in enumerate(self.f_sizes)
        ]
        
        self.out = tf.keras.layers.Dense(2, activation='softmax', kernel_initializer=tf.keras.initializers.HeUniform(), name='model-output')
                
    def call(self, p):
        if self.is_evefn:
            z, p = p
        if self.is_evpfnid:
            p = self.encoder(p)
        for ppm in self.ppm_layers:
            p = ppm(p)
        for equi in self.equi_layers:
            p = equi([p,z] if self.is_evefn else p)
        p = self.proj(self.z_weighting([p,z]) if self.is_evefn else p)
        for d, F in zip(self.dropout_layers, self.F_layers):
            p = d(F(p))
        p = self.out(p)
            
        return p