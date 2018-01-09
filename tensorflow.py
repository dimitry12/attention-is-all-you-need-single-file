def multiattention_head_with_projections(Q_source,
                                         K_source,
                                         V_source,
                                         multi_x8 = 1,
                                         keys_queries_projection_dim_x8 = 1,
                                         values_projection_dim_x8 = 1):
    
    # batch-size should be the same
    # assert tf.shape(Q_source)[0] == tf.shape(K_source)[0] == tf.shape(V_source)[0]
    
    # don't need Q and K to have the same dimensionality,
    # because we project them into the same dimensionality
    # assert tf.shape(Q_source)[-1] ? tf.shape(K_source)[-1]
    
    # don't need the same number of Q and K: we can use few queries
    # to select from large vocabulary
    
    # we should have the same number of keys and values to attend to
    # assert tf.shape(K_source)[1] == tf.shape(V_source)[1]
    
    # All variables have dimensions independent from dict_length,
    # which is only used for reshaping here. This makes the transformer
    # useful for variable length sequences. The number of parameters in the
    # model is independent from sequence length. This is key, and was
    # not obvious to me.
    batch_size = tf.shape(V_source)[0]
    dict_length = tf.shape(V_source)[1]
    input_channels = V_source.get_shape().as_list()[-1] # we only use this to reproject output-values to input-values dimensions
    
    with tf.variable_scope('{}head_selfattention_with_{}projected_keys_and_{}projected_values'. format(multi_x8*8, keys_queries_projection_dim_x8*8, values_projection_dim_x8*8)):
        
        projection_coeffs_initializer = tf.ones_initializer()
        projection_coeffs_initializer = tf.random_uniform_initializer()
        
        with tf.name_scope('initialize_projection_coeffs'):            
        
            keys_projection_coeffs = tf.get_variable('keys_projection_coeffs',
                                               shape=(1,1,K_source.get_shape().as_list()[-1],keys_queries_projection_dim_x8*8*multi_x8*8),
                                               initializer=projection_coeffs_initializer,
                                               dtype=hparams['dtype'])
            queries_projection_coeffs = tf.get_variable('queries_projection_coeff',
                                               shape=(1,1,Q_source.get_shape().as_list()[-1],keys_queries_projection_dim_x8*8*multi_x8*8),
                                               initializer=projection_coeffs_initializer,
                                               dtype=hparams['dtype'])
            values_projection_coeffs = tf.get_variable('values_projection_coeff',
                                               shape=(1,1,input_channels,values_projection_dim_x8*8*multi_x8*8),
                                               initializer=projection_coeffs_initializer,
                                               dtype=hparams['dtype'])
        
        with tf.name_scope('projecting'):
            
            
            def _project(tensor_to_project, coeffs, name):
                tensor_to_project_expanded = tf.expand_dims(tensor_to_project, 1) # [batch_size x 1 x seq_length x channels]
                return tf.transpose(tf.reshape(tf.nn.conv2d(tensor_to_project_expanded , coeffs,
                                                            strides=[1,1,1,1],
                                                            data_format='NHWC',
                                                            padding='VALID'),
                                               shape=(batch_size,tf.shape(tensor_to_project)[1],multi_x8*8,-1)),
                                    perm=[0,2,1,3],
                                    name='{}_projecting'.format(name)) # batch x heads x seq x channels

            keys_projections = _project(K_source, keys_projection_coeffs, 'keys')
            queries_projections = _project(Q_source, queries_projection_coeffs, 'queries')
            values_projections = _project(V_source, values_projection_coeffs, 'values')
                        
        with tf.name_scope('calculating_attention'):
            # softmax is nonlinearity
            attention_softmaxed = tf.cast(tf.nn.softmax(tf.cast(tf.divide(tf.matmul(queries_projections, tf.transpose(keys_projections, perm=[0,1,3,2])), # batch x heads x seq x seq
                                                                          tf.sqrt(tf.cast(keys_queries_projection_dim_x8*8, hparams['dtype']))), tf.float32)),
                                          hparams['dtype'],
                                          name='attention_softmaxed') # batch x heads x seq x seq
        
        
        # attention does the sum of features, works best for features in linear spaces
        projected_multiselfattended_input = tf.matmul(attention_softmaxed, values_projections,
                                                      name='projected_multiselfattended_input') # batch x heads x seq x values_channels
        
        with tf.name_scope('initialize_reprojection_coeffs'):                    
            reprojection_coeffs = tf.get_variable('reprojection_coeffs',
                                                  shape=(1,1,multi_x8*8*values_projection_dim_x8*8, input_channels),
                                                  initializer=projection_coeffs_initializer,
                                                  dtype=hparams['dtype'])
            
        with tf.name_scope('reprojecting'):    
            selfattended_input = tf.reshape(tf.nn.conv2d(tf.reshape(tf.transpose(projected_multiselfattended_input,
                                                                                 perm=[0,2,1,3]),
                                                                    shape=(batch_size, 1, dict_length, -1)),
                                                         reprojection_coeffs,
                                                         strides=[1,1,1,1],
                                                         data_format='NHWC',
                                                         padding='VALID'),
                                            shape=(batch_size,dict_length,input_channels),
                                            name='selfattended_input')
                          
        return selfattended_input
        
def ff(input_tensor, ff_projection_dim_x8 = None):
    batch_size = tf.shape(input_tensor)[0]
    seq_length = tf.shape(input_tensor)[1]
    input_channels = input_tensor.get_shape().as_list()[-1]
    
    
    if ff_projection_dim_x8 is None:
        ff_projection_dim_x8 = input_channels // 8 # default to same number of channels as input
        
    with tf.variable_scope('ff_sublayer'):
        projection_coeffs_initializer = tf.ones_initializer()
        projection_coeffs_initializer = tf.random_uniform_initializer()
    
        ff_project_kernel = tf.get_variable('ff_project_kernel',
                                            shape=(1,1,input_channels,ff_projection_dim_x8*8),
                                            initializer=projection_coeffs_initializer,
                                            dtype=hparams['dtype'])
        ff_project_bias = tf.get_variable('ff_project_bias',
                                            shape=(ff_projection_dim_x8*8),
                                            initializer=projection_coeffs_initializer,
                                            dtype=hparams['dtype'])
        ff_reproject_kernel = tf.get_variable('ff_reproject_kernel',
                                              shape=(1,1,ff_projection_dim_x8*8,input_channels),
                                              initializer=projection_coeffs_initializer,
                                              dtype=hparams['dtype'])
        
        output = tf.reshape(tf.nn.conv2d(tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(tf.reshape(input_tensor,
                                                                                           shape=(batch_size, 1, seq_length, input_channels)),
                                                                                ff_project_kernel,
                                                                                strides=[1,1,1,1],
                                                                                padding='VALID'),
                                                                   ff_project_bias)),
                                         ff_reproject_kernel,
                                         strides=[1,1,1,1],
                                         padding='VALID'),
                            shape=(batch_size,seq_length,input_channels),
                            name='ff_double_conved')
                            
    return output
        
def encoder_layer(input_tensor,
                  multi_x8=1,
                  keys_queries_projection_dim_x8=1,
                  values_projection_dim_x8=1,
                  ff_projection_dim_x8=None):
    enc_attention = multiattention_head_with_projections(input_tensor,
                                                         input_tensor,
                                                         input_tensor,
                                                         multi_x8=multi_x8,
                                                         keys_queries_projection_dim_x8=keys_queries_projection_dim_x8,
                                                         values_projection_dim_x8=values_projection_dim_x8) 

    attention_with_res = tf.add(input_tensor,enc_attention,name='attention_with_res')
        
    normed_attention_with_res = tf.contrib.layers.layer_norm(attention_with_res)
                            
    ff_with_res = tf.add(ff(normed_attention_with_res, ff_projection_dim_x8),
                         normed_attention_with_res,
                         name='ff_with_res')
        
    normed_ff_with_res = tf.contrib.layers.layer_norm(ff_with_res)
        
    return normed_ff_with_res

def decoder_layer(input_tensor,
                  encoder_output,
                  multi_x8=1,
                  keys_queries_projection_dim_x8=1,
                  values_projection_dim_x8=1,
                  ff_projection_dim_x8=None):
    dec_attention = multiattention_head_with_projections(input_tensor,
                                                         input_tensor,
                                                         input_tensor,
                                                         multi_x8=multi_x8,
                                                         keys_queries_projection_dim_x8=keys_queries_projection_dim_x8,
                                                         values_projection_dim_x8=values_projection_dim_x8) 

    dec_attention_with_res = tf.add(input_tensor,dec_attention,name='dec_attention_with_res')
        
    normed_dec_attention_with_res = tf.contrib.layers.layer_norm(dec_attention_with_res)
    
    with tf.variable_scope('decoder_querying_encoder'):
        enc_attention = multiattention_head_with_projections(normed_dec_attention_with_res,
                                                             encoder_output,
                                                             encoder_output,
                                                             multi_x8=multi_x8,
                                                             keys_queries_projection_dim_x8=keys_queries_projection_dim_x8,
                                                             values_projection_dim_x8=values_projection_dim_x8) 

    enc_attention_with_res = tf.add(normed_dec_attention_with_res,enc_attention,name='enc_attention_with_res')
        
    normed_enc_attention_with_res = tf.contrib.layers.layer_norm(enc_attention_with_res)
                            
    ff_with_res = tf.add(ff(normed_enc_attention_with_res, ff_projection_dim_x8),
                         normed_enc_attention_with_res,
                         name='ff_with_res')
        
    normed_ff_with_res = tf.contrib.layers.layer_norm(ff_with_res)
        
    return normed_ff_with_res
    
hparams = {
    'dtype': tf.float16,
    'batch_size': 72,
}
