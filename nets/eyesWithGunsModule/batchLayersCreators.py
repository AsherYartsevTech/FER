from GlobalVarsAndLibs import *
from InitializingLayersDicts import *

def bnInitFeed(gamma, beta):
    '''
    purpose:
    helper function which decides wheather init values will propagate from a previous stage 
    (the corresponding layer in previous stage)
    or no initial values will propagate, thus the function that calls it will initiate on its own default values
    
    way:
    program flow is designed that way that only in case that we want to propagate weights from previous stage ,
    gamma and beta params will differ from None
    '''
    if None == gamma or None == beta:
        return None
    return {'gamma' : gamma,
            'beta'  : beta,
            'moving_mean' : tf.constant_initializer(0.),
            'moving_variance' : tf.constant_initializer(1.)}


def CreateEyeLayers(    self, 
                        input_image, 
                        output_dim,
                        init_dict,
                        scope='NoEyeScopeProvided', 
                        optimizer=tf.train.AdamOptimizer, 
                        lr=0.01, 
                        filter_sizes = {'firstFilter' : 5, 'commonFilter': 3}, 
                        num_filters={'outChannel1' : 64, 'outChannel2': 128, 'outChannel3' : 256},
                        trainable=True):
    '''
    purpose:
    creates layers in a design which corresponds to our Eye design (which is basically a vgg style graph).

    '''
    '''first convolution series'''
    layers = conv2d(    input_image,
                        num_outputs=num_filters['outChannel1'],
                        weights_initializer=init_dict['conv1/weights'],
                        biases_initializer=init_dict['conv1/biases'],
                        trainable=trainable,
                        activation_fn=None,
                        kernel_size=filter_sizes['firstFilter'],
                        stride=[1,1], 
                        padding='SAME',
                        scope="%s/conv1" %(scope),
                        variables_collections=tf.GraphKeys.TRAINABLE_VARIABLES)
    layers   = batch_norm(  layers,
                            center=True,
                            scale=True,
                            param_initializers=bnInitFeed(init_dict['batchNorm1/gamma'],
                                                          init_dict['batchNorm1/beta']),
                            activation_fn=tf.nn.relu,
                            updates_collections=tf.GraphKeys.UPDATE_OPS,
                            is_training=self.is_training,
                            trainable=trainable,
                            batch_weights=None,
                            scope='%s/batchNorm1' %(scope))
    layers = conv2d(    layers,
                        num_outputs=num_filters['outChannel1'],
                        trainable=trainable,
                        weights_initializer=init_dict['conv2/weights'],
                        biases_initializer=init_dict['conv2/biases'],
                        activation_fn=None,
                        kernel_size=filter_sizes['commonFilter'],
                        stride=[1,1], 
                        padding='SAME',
                        scope="%s/conv2" %(scope),
                        variables_collections=tf.GraphKeys.TRAINABLE_VARIABLES)
    layers   = batch_norm(  layers,
                            center=True,
                            scale=True,
                            param_initializers=bnInitFeed(init_dict['batchNorm2/gamma'],
                                                          init_dict['batchNorm2/beta']),
                            activation_fn=tf.nn.relu,
                            updates_collections=tf.GraphKeys.UPDATE_OPS,
                            is_training=self.is_training,
                            trainable=trainable,
                            batch_weights=None,
                            scope='%s/batchNorm2' %(scope))
    layers = max_pool2d(    layers,
                            kernel_size=3,
                            stride=2,
                            padding='SAME',
                            scope='%s/maxpool1' %(scope))
    layers = dropout(layers, keep_prob = 0.25,is_training=self.is_training,scope='%s/dropout1' %(scope))
    
    
    '''2nd convolution series'''
    layers = conv2d(layers,
                        num_outputs=num_filters['outChannel2'],
                        trainable=trainable,
                        weights_initializer=init_dict['conv3/weights'],
                        biases_initializer= init_dict['conv3/biases'],
                        activation_fn=None,
                        kernel_size=filter_sizes['commonFilter'],
                        stride=[1,1], 
                        padding='SAME',
                        scope="%s/conv3" %(scope),
                        variables_collections=tf.GraphKeys.TRAINABLE_VARIABLES)
    layers   = batch_norm(  layers,
                            center=True,
                            scale=True,
                            param_initializers=bnInitFeed(init_dict['batchNorm3/gamma'],
                                                          init_dict['batchNorm3/beta']),
                            activation_fn=tf.nn.relu,
                            updates_collections=tf.GraphKeys.UPDATE_OPS,
                            is_training=self.is_training,
                            trainable=trainable,
                            batch_weights=None,
                            scope='%s/batchNorm3' %(scope))
    layers = conv2d(    layers,
                        num_outputs=num_filters['outChannel2'],
                        trainable=trainable,
                        weights_initializer=init_dict['conv4/weights'],
                        biases_initializer= init_dict['conv4/biases'],
                        activation_fn=None,
                        kernel_size=filter_sizes['commonFilter'],
                        stride=[1,1], 
                        padding='SAME',
                        scope="%s/conv4" %(scope),
                        variables_collections=tf.GraphKeys.TRAINABLE_VARIABLES)
    layers   = batch_norm(  layers,
                            center=True,
                            scale=True,
                            param_initializers=bnInitFeed(init_dict['batchNorm4/gamma'],
                                                          init_dict['batchNorm4/beta']),
                            activation_fn=tf.nn.relu,
                            updates_collections=tf.GraphKeys.UPDATE_OPS,
                            is_training=self.is_training,
                            trainable=trainable,
                            batch_weights=None,
                            scope='%s/batchNorm4' %(scope))
    layers = max_pool2d(    layers,
                            kernel_size=3,
                            stride=2,
                            padding='SAME',
                            scope='%s/maxpool2' %(scope))
    layers = dropout(layers, keep_prob = 0.25,is_training=self.is_training,scope='%s/dropout2' %(scope))

    '''3rd convolution deries'''
    layers = conv2d(layers,
                        num_outputs=num_filters['outChannel3'],
                        trainable=trainable,
                        weights_initializer=init_dict['conv5/weights'],
                        biases_initializer= init_dict['conv5/biases'],
                        activation_fn=None,
                        kernel_size=filter_sizes['commonFilter'],
                        stride=[1,1], 
                        padding='SAME',
                        scope="%s/conv5" %(scope),
                        variables_collections=tf.GraphKeys.TRAINABLE_VARIABLES)
    layers   = batch_norm(  layers,
                            center=True,
                            scale=True,
                            param_initializers=bnInitFeed(init_dict['batchNorm5/gamma'],
                                                          init_dict['batchNorm5/beta']),
                            activation_fn=tf.nn.relu,
                            updates_collections=tf.GraphKeys.UPDATE_OPS,
                            is_training=self.is_training,
                            trainable=trainable,
                            batch_weights=None,
                            scope='%s/batchNorm5' %(scope))
    layers = conv2d(layers,
                        num_outputs=num_filters['outChannel3'],
                        trainable=trainable,
                        weights_initializer=init_dict['conv6/weights'],
                        biases_initializer= init_dict['conv6/biases'],
                        activation_fn=None,
                        kernel_size=filter_sizes['commonFilter'],
                        stride=[1,1], 
                        padding='SAME',
                        scope="%s/conv6" %(scope),
                        variables_collections=tf.GraphKeys.TRAINABLE_VARIABLES)
    layers   = batch_norm(  layers,
                            center=True,
                            scale=True,
                            param_initializers=bnInitFeed(init_dict['batchNorm6/gamma'],
                                                          init_dict['batchNorm6/beta']),
                            activation_fn=tf.nn.relu,
                            updates_collections=tf.GraphKeys.UPDATE_OPS,
                            is_training=self.is_training,
                            trainable=trainable,
                            batch_weights=None,
                            scope='%s/batchNorm6' %(scope))
    layers = conv2d(    layers,
                        num_outputs=num_filters['outChannel3'],
                        trainable=trainable,
                        weights_initializer=init_dict['conv7/weights'],
                        biases_initializer= init_dict['conv7/biases'],
                        activation_fn=None,
                        kernel_size=filter_sizes['commonFilter'],
                        stride=[1,1], 
                        padding='SAME',
                        scope="%s/conv7" %(scope),
                        variables_collections=tf.GraphKeys.TRAINABLE_VARIABLES)
    layers   = batch_norm(  layers,
                            center=True,
                            scale=True,
                            param_initializers=bnInitFeed(init_dict['batchNorm7/gamma'],
                                                          init_dict['batchNorm7/beta']),
                            activation_fn=tf.nn.relu,
                            updates_collections=tf.GraphKeys.UPDATE_OPS,
                            is_training=self.is_training,
                            trainable=trainable,
                            batch_weights=None,
                            scope='%s/batchNorm7' %(scope))
    layers = max_pool2d(    layers,
                            kernel_size=3,
                            stride=2,
                            padding='SAME',
                            scope='%s/maxpool3' %(scope))
    layers = dropout(layers, keep_prob = 0.25,is_training=self.is_training,scope='%s/dropout3' %(scope))
    
    '''4th convolution deries'''
    layers = conv2d(layers,
                        num_outputs=num_filters['outChannel3'],
                        trainable=trainable,
                        weights_initializer=init_dict['conv8/weights'],
                        biases_initializer= init_dict['conv8/biases'],
                        activation_fn=None,
                        kernel_size=filter_sizes['commonFilter'],
                        stride=[1,1], 
                        padding='SAME',
                        scope="%s/conv8" %(scope),
                        variables_collections=tf.GraphKeys.TRAINABLE_VARIABLES)
    layers   = batch_norm(  layers,
                            center=True,
                            scale=True,
                            param_initializers=bnInitFeed(init_dict['batchNorm8/gamma'],
                                                          init_dict['batchNorm8/beta']),
                            activation_fn=tf.nn.relu,
                            updates_collections=tf.GraphKeys.UPDATE_OPS,
                            is_training=self.is_training,
                            trainable=trainable,
                            batch_weights=None,
                            scope='%s/batchNorm8' %(scope))
    layers = conv2d(layers,
                        num_outputs=num_filters['outChannel3'],
                        trainable=trainable,
                        weights_initializer=init_dict['conv9/weights'],
                        biases_initializer= init_dict['conv9/biases'],
                        activation_fn=None,
                        kernel_size=filter_sizes['commonFilter'],
                        stride=[1,1], 
                        padding='SAME',
                        scope="%s/conv9" %(scope),
                        variables_collections=tf.GraphKeys.TRAINABLE_VARIABLES)
    layers   = batch_norm(  layers,
                            center=True,
                            scale=True,
                            param_initializers=bnInitFeed(init_dict['batchNorm9/gamma'],
                                                          init_dict['batchNorm9/beta']),
                            activation_fn=tf.nn.relu,
                            updates_collections=tf.GraphKeys.UPDATE_OPS,
                            is_training=self.is_training,
                            trainable=trainable,
                            batch_weights=None,
                            scope='%s/batchNorm9' %(scope))
    layers = conv2d(    layers,
                        num_outputs=num_filters['outChannel3'],
                        trainable=trainable,
                        weights_initializer=init_dict['conv10/weights'],
                        biases_initializer= init_dict['conv10/biases'],
                        activation_fn=None,
                        kernel_size=filter_sizes['commonFilter'],
                        stride=[1,1], 
                        padding='SAME',
                        scope="%s/conv10" %(scope),
                        variables_collections=tf.GraphKeys.TRAINABLE_VARIABLES)
    layers   = batch_norm(  layers,
                            center=True,
                            scale=True,
                            param_initializers=bnInitFeed(init_dict['batchNorm10/gamma'],
                                                          init_dict['batchNorm10/beta']),
                            activation_fn=tf.nn.relu,
                            updates_collections=tf.GraphKeys.UPDATE_OPS,
                            is_training=self.is_training,
                            trainable=trainable,
                            batch_weights=None,
                            scope='%s/batchNorm10' %(scope))
    maxpoolLastConv = max_pool2d(layers,
                            kernel_size=3,
                            stride=2,
                            padding='SAME',
                            scope='%s/maxpool4' %(scope))
    layers = dropout(maxpoolLastConv, keep_prob = 0.25,is_training=self.is_training,scope='%s/dropout4' %(scope))
    '''fully connected section'''
    layers = flatten(layers,
                        scope='%s/flatten' % (scope))
    layers = fully_connected(inputs=layers, 
                                    num_outputs=1024,
                                    trainable=trainable,
                                    weights_initializer=init_dict['dense1/weights'],
                                    biases_initializer= init_dict['dense1/biases'],
                                    activation_fn=None,
                                    variables_collections=tf.GraphKeys.TRAINABLE_VARIABLES,
                                    scope='%s/dense1' % (scope))
    layers   = batch_norm(  layers,
                            center=True,
                            scale=True,
                            param_initializers=bnInitFeed(init_dict['batchNorm11/gamma'],
                                                          init_dict['batchNorm11/beta']),
                            activation_fn=tf.nn.relu,
                            updates_collections=tf.GraphKeys.UPDATE_OPS,
                            is_training=self.is_training,
                            trainable=trainable,
                            batch_weights=None,
                            scope='%s/batchNorm11' %(scope))
    layers = dropout(layers, keep_prob = 0.25,
                        is_training=self.is_training,scope='%s/dropout5' %(scope))
    layers = fully_connected(inputs=layers, 
                                    num_outputs=1024,
                                    trainable=trainable,
                                    weights_initializer=init_dict['dense2/weights'],
                                    biases_initializer= init_dict['dense2/biases'],
                                    activation_fn=None,
                                    variables_collections=tf.GraphKeys.TRAINABLE_VARIABLES,
                                    scope='%s/dense2' % (scope))
    layers   = batch_norm(  layers,
                            center=True,
                            scale=True,
                            param_initializers=bnInitFeed(init_dict['batchNorm12/gamma'],
                                                          init_dict['batchNorm12/beta']),
                            activation_fn=tf.nn.relu,
                            updates_collections=tf.GraphKeys.UPDATE_OPS,
                            is_training=self.is_training,
                            trainable=trainable,
                            batch_weights=None,
                            scope='%s/batchNorm12' %(scope))
    layers = dropout(layers, keep_prob = 0.25,
                        is_training=self.is_training,scope='%s/dropout6' %(scope))

    classify = fully_connected(inputs=layers, 
                                    num_outputs=output_dim,
                                    trainable=trainable,
                                    weights_initializer=init_dict['classify/weights'],
                                    biases_initializer= init_dict['classify/biases'],
                                    activation_fn=None,
                                    variables_collections=tf.GraphKeys.TRAINABLE_VARIABLES,
                                    scope='%s/classify' % (scope))
    return classify,maxpoolLastConv


def CreateBrainLayers(          self, 
                                eye1_input, 
                                eye2_input,
                                output_dim=7,
                                scope='NoEyeScopeProvided', 
                                optimizer=tf.train.AdamOptimizer, 
                                lr=0.01, 
                                filter_sizes = {'firstFilter' : 5, 'commonFilter': 3}, 
                                num_filters={'outChannel1' : 64, 'outChannel2': 128, 'outChannel3' : 256},
                                init_dict=BrainInitDictStage1()()):
    '''
    purpose:
    
    create a series of fully connected layers which get as an input a concatenated feature-outputs of both eyes and outputs
    a classification
    '''
    
    
    '''fully connected section'''
    eye1_flattened = flatten(eye1_input,
                        scope='%s/flattenEye1' % (scope))
    eye2_flattened = flatten(eye2_input,
                        scope='%s/flattenEye2' % (scope))
    merged_eyes = tf.concat([eye1_flattened,eye2_flattened], axis=1,name='%s/concatEyes' % (scope))

    layers = fully_connected(inputs=merged_eyes, 
                                    num_outputs=1024,
                                    weights_initializer=init_dict['dense1/weights'],
                                    biases_initializer= init_dict['dense1/biases'],
                                    activation_fn=None,
                                    variables_collections=tf.GraphKeys.TRAINABLE_VARIABLES,
                                    scope='%s/dense1' % (scope))
    layers   = batch_norm(  layers,
                            center=True,
                            scale=True,
                            param_initializers=bnInitFeed(init_dict['batchNorm1/gamma'],
                                                          init_dict['batchNorm1/beta']),
                            activation_fn=tf.nn.relu,
                            updates_collections=tf.GraphKeys.UPDATE_OPS,
                            is_training=self.is_training,
                            trainable=True,
                            batch_weights=None,
                            scope='%s/batchNorm1' %(scope))
    layers = dropout(layers, keep_prob = 0.25,
                        is_training=self.is_training,scope='%s/dropout1' %(scope))

    layers = fully_connected(inputs=layers, 
                                    num_outputs=1024,
                                    weights_initializer=init_dict['dense2/weights'],
                                    biases_initializer= init_dict['dense2/biases'],
                                    activation_fn=None,
                                    variables_collections=tf.GraphKeys.TRAINABLE_VARIABLES,
                                    scope='%s/dense2' % (scope))
    layers   = batch_norm(  layers,
                            center=True,
                            scale=True,
                            param_initializers=bnInitFeed(init_dict['batchNorm2/gamma'],
                                                          init_dict['batchNorm2/beta']),
                            activation_fn=tf.nn.relu,
                            updates_collections=tf.GraphKeys.UPDATE_OPS,
                            is_training=self.is_training,
                            trainable=True,
                            batch_weights=None,
                            scope='%s/batchNorm2' %(scope))
    layers = dropout(layers, keep_prob = 0.25,
                        is_training=self.is_training,scope='%s/dropout2' %(scope))

    layers = fully_connected(inputs=layers, 
                                    num_outputs=1024,
                                    weights_initializer=init_dict['dense3/weights'],
                                    biases_initializer= init_dict['dense3/biases'],
                                    activation_fn=None,
                                    variables_collections=tf.GraphKeys.TRAINABLE_VARIABLES,
                                    scope='%s/dense3' % (scope))
    layers   = batch_norm(  layers,
                            center=True,
                            scale=True,
                            param_initializers=bnInitFeed(init_dict['batchNorm3/gamma'],
                                                          init_dict['batchNorm3/beta']),
                            activation_fn=tf.nn.relu,
                            updates_collections=tf.GraphKeys.UPDATE_OPS,
                            is_training=self.is_training,
                            trainable=True,
                            batch_weights=None,
                            scope='%s/batchNorm3' %(scope))
    layers = dropout(layers, keep_prob = 0.25,
                        is_training=self.is_training,scope='%s/dropout3' %(scope))
    
    #no activation function in last layer since activation is applied in SpicyLoss where it is used
    brain_classify = fully_connected(inputs=layers, 
                                    num_outputs=output_dim,
                                    weights_initializer=init_dict['classify/weights'],
                                    biases_initializer= init_dict['classify/biases'],
                                    activation_fn=None,
                                    variables_collections=tf.GraphKeys.TRAINABLE_VARIABLES,                                         
                                    scope='%s/classify' % (scope))
    return brain_classify


def CreateDescreminatorLayers(  self, 
                                input_image, 
                                isFalseEye,
                                output_dim=1,
                                scope='NoEyeScopeProvided', 
                                optimizer=tf.train.AdamOptimizer, 
                                lr=0.01, 
                                filter_sizes = {'firstFilter' : 5, 'commonFilter': 3}, 
                                num_filters={'outChannel1' : 64, 'outChannel2': 128, 'outChannel3' : 256},
                                init_dict=DescInitDictStage1()()):

    '''
    purpose:
    create a descriminator-like component in the net which converges to 1 for eye1 outputs on images and converges
    to 0 on eye2 outputs on images.
    '''

    '''first convolution series'''
    layers = conv2d(input_image,
                        num_outputs=num_filters['outChannel1'],
                        weights_initializer=init_dict['conv1/weights'],
                        biases_initializer= init_dict['conv1/biases'],
                        activation_fn=None,
                        kernel_size=filter_sizes['firstFilter'],
                        stride=[1,1], 
                        padding='SAME',
                        scope="%s/conv1" %(scope),
                        variables_collections=tf.GraphKeys.TRAINABLE_VARIABLES)
    layers   = batch_norm(  layers,
                            center=True,
                            scale=True,
                            param_initializers=bnInitFeed(init_dict['batchNorm1/gamma'],
                                                          init_dict['batchNorm1/beta']),
                            activation_fn=tf.nn.relu,
                            updates_collections=tf.GraphKeys.UPDATE_OPS,
                            is_training=self.is_training,
                            trainable=True,
                            batch_weights=None,
                            scope='%s/batchNorm1' %(scope))
    layers = conv2d(    layers,
                        num_outputs=num_filters['outChannel1'],
                        weights_initializer=init_dict['conv2/weights'],
                        biases_initializer= init_dict['conv2/biases'],
                        activation_fn=None,
                        kernel_size=filter_sizes['commonFilter'],
                        stride=[1,1], 
                        padding='SAME',
                        scope="%s/conv2" %(scope),
                        variables_collections=tf.GraphKeys.TRAINABLE_VARIABLES)
    layers   = batch_norm(  layers,
                            center=True,
                            scale=True,
                            param_initializers=bnInitFeed(init_dict['batchNorm2/gamma'],
                                                          init_dict['batchNorm2/beta']),
                            activation_fn=tf.nn.relu,
                            updates_collections=tf.GraphKeys.UPDATE_OPS,
                            is_training=self.is_training,
                            trainable=True,
                            batch_weights=None,
                            scope='%s/batchNorm2' %(scope))
    layers = max_pool2d(    layers,
                            kernel_size=3,
                            stride=2,
                            padding='SAME',
                            scope='%s/maxpool1' %(scope))
    layers = dropout(layers, keep_prob = 0.25,is_training=self.is_training,scope='%s/dropout1' %(scope))
    
    '''fully connected section'''
    layers = flatten(layers,
                        scope='%s/flatten' % (scope))
    layers = fully_connected(inputs=layers, 
                                    num_outputs=1024,
                                    weights_initializer=init_dict['dense1/weights'],
                                    biases_initializer= init_dict['dense1/biases'],
                                    activation_fn=None,
                                    variables_collections=tf.GraphKeys.TRAINABLE_VARIABLES,
                                    scope='%s/dense1' % (scope))
    layers   = batch_norm(  layers,
                            center=True,
                            scale=True,
                            param_initializers=bnInitFeed(init_dict['batchNorm3/gamma'],
                                                          init_dict['batchNorm3/beta']),
                            activation_fn=tf.nn.relu,
                            updates_collections=tf.GraphKeys.UPDATE_OPS,
                            is_training=self.is_training,
                            trainable=True,
                            batch_weights=None,
                            scope='%s/batchNorm3' %(scope))
    layers = dropout(layers, keep_prob = 0.25,
                        is_training=self.is_training,scope='%s/dropout5' %(scope))
    layers = fully_connected(inputs=layers, 
                                    num_outputs=1024,
                                    weights_initializer=init_dict['dense2/weights'],
                                    biases_initializer= init_dict['dense2/biases'],
                                    activation_fn=None,
                                    variables_collections=tf.GraphKeys.TRAINABLE_VARIABLES,
                                    scope='%s/dense2' % (scope))
    layers   = batch_norm(  layers,
                            center=True,
                            scale=True,
                            param_initializers=bnInitFeed(init_dict['batchNorm4/gamma'],
                                                          init_dict['batchNorm4/beta']),
                            activation_fn=tf.nn.relu,
                            updates_collections=tf.GraphKeys.UPDATE_OPS,
                            is_training=self.is_training,
                            trainable=True,
                            batch_weights=None,
                            scope='%s/batchNorm4' %(scope))
    layers = dropout(layers, keep_prob = 0.25,
                        is_training=self.is_training,scope='%s/dropout6' %(scope))
    if False == isFalseEye:
        classify = fully_connected(inputs=layers, 
                                    num_outputs=output_dim,
                                    weights_initializer=init_dict['TrueProbability/weights'],
                                    biases_initializer= init_dict['TrueProbability/biases'],
                                    activation_fn=None,
                                    variables_collections=tf.GraphKeys.TRAINABLE_VARIABLES,                                         
                                    scope='%s/TrueProbability' % (scope))
    else:
        classify = fully_connected(inputs=layers, 
                                    num_outputs=output_dim,
                                    weights_initializer=init_dict['FalseProbability/weights'],
                                    biases_initializer= init_dict['FalseProbability/biases'],
                                    activation_fn=None,
                                    variables_collections=tf.GraphKeys.TRAINABLE_VARIABLES,                                         
                                    scope='%s/FalseProbability' % (scope))
    return classify

