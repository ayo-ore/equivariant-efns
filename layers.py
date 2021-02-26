import keras
import keras.backend as K

class Equivariant(keras.layers.Layer):
    def __init__(self, output_dim, variation='bias', max_particles=139, activation='relu', lambda_zero=False, gamma_zero=False, **kwargs):
        self.output_dim = output_dim
        self.var = variation
        self.mp = max_particles
        self.activation = keras.activations.get(activation)
        self.lambda_zero = lambda_zero
        self.gamma_zero = gamma_zero
        self.supports_masking = True
        super(Equivariant, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.var in ['sum','max']:
            self.lam = self.add_weight(name='lambda',
                                       shape=(input_shape[-1], self.output_dim),
                                       initializer = 'zeros' if self.lambda_zero else 'he_uniform',
                                       trainable = False if self.lambda_zero else True)
        self.gam = self.add_weight(name='gamma',
                                   shape=(input_shape[-1], self.output_dim),
                                   initializer = 'zeros' if self.gamma_zero else 'he_uniform',
                                   trainable = False if self.gamma_zero else True)
        if self.var in ['bias']:
            self.bias = self.add_weight(name='bias',
                                        shape = (self.output_dim,),
                                        initializer=keras.initializers.Constant(0.5),
                                        trainable=True)
            self.bias = K.expand_dims(self.bias, axis=-1)
            self.bias = K.repeat(self.bias, self.mp)
            self.bias = K.squeeze(self.bias, axis=-1)
            self.bias = K.permute_dimensions(self.bias, (1,0))
        super(Equivariant, self).build(input_shape)

    def compute_output_shape(self, input_shape):
            return (None, input_shape[-2], self.output_dim)

    def call(self, x, mask=None):
        if self.var == 'sum':
            """
            f(x) = sigma( x lam  -  1 1^T x gam)
            use self.gam and self.lam only
            """
            M = self.mp
            # Lmat = K.dot(x, self.lam)
            # Rmat = K.permute_dimensions(K.dot(K.ones((M,M)), K.dot(x, self.gam)), (1,0,2))
            output = K.dot(x, self.lam) - K.permute_dimensions(K.dot(K.ones((M,M)), K.dot(x, self.gam)), (1,0,2))
            if self.activation != None:
                output = self.activation(output)
            return output

        if self.var == 'max':
            """
            f(x) = sigma( x lam  -  1 maxpool(x) gam)
            use self.gam and self.lam only
            """
            M = self.mp
            Lmat = K.dot(x, self.lam)
            maxgam = K.dot(K.max(x,axis=-2, keepdims=True), self.gam)
            Rmat = K.permute_dimensions(K.dot(K.ones((M,1)), maxgam), (1,0,2))
            output = Lmat - Rmat
            if self.activation != None:
                output = self.activation(output)
            return output

        if self.var == 'bias':
            """
            f(x) = sigma(bias  +  (x  -  1 maxpool(x)) gam)
            use self.gam and self.bias only
            """
            M = self.mp
            onemax = K.permute_dimensions(K.dot(K.ones((M,1)), K.max(x,axis=-2,keepdims=True)), (1,0,2))
            Lmat = x - onemax
            output = K.dot(Lmat, self.gam)
            output = K.bias_add(output, self.bias, data_format='channels_last')
            if self.activation != None:
                output = self.activation(output)
            return output


class IRCEquivariant(keras.layers.Layer):
    def __init__(self, output_dim, max_particles=139, activation='relu', variation=None, lambda_zero=False, gamma_zero=False, **kwargs):
        self.output_dim = output_dim
        self.mp = max_particles
        self.activation = keras.activations.get(activation)
        self.lambda_zero = lambda_zero
        self.gamma_zero = gamma_zero
        super(IRCEquivariant, self).__init__(**kwargs)

    def build(self, input_shape):
        self.lam = self.add_weight(name='lambda',
                                   shape=(input_shape[0][-1], self.output_dim),
                                   initializer = 'zeros' if self.lambda_zero else 'he_uniform',
                                   trainable = False if self.lambda_zero else True)
        self.gam = self.add_weight(name='gamma',
                                   shape=(input_shape[0][-1], self.output_dim),
                                   initializer = 'zeros' if self.gamma_zero else 'he_uniform',
                                   trainable = False if self.gamma_zero else True)
        super(IRCEquivariant, self).build(input_shape)

    def compute_output_shape(self, input_shape):
            return (None, input_shape[0][-2], self.output_dim)

    def call(self, x, mask=None):
        """
        f(x) = sigma( x lam  -  1 z^T x gam)
        use self.gam and self.lam only
        """
        x, w = x
        M = self.mp
        Lmat = K.dot(x, self.lam)
        xgam = K.dot(x, self.gam)
        wxgam = K.batch_dot(w,xgam,1)
        Rmat = K.repeat(wxgam,M)
        output = Lmat - Rmat
        if self.activation != None:
            output = self.activation(output)
        return output


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
     
#     @tf.function
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