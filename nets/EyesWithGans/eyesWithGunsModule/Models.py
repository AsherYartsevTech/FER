from GlobalVarsAndLibs import *
from Utils import *
from enum import Enum
from InitializingLayersDicts import *
from batchLayersCreators import *



class Model_EWG_stage1:
    '''
    EWG stands for Eyes With GANs
    '''
    
    def __init__(self, 
                name, 
                input_dim, 
                output_dim,
                optimizer=tf.train.AdamOptimizer, 
                lr=0.01, 
                filter_sizes = {'firstFilter' : 5, 'commonFilter': 3}, 
                num_filters={'outChannel1' : 64, 'outChannel2': 128, 'outChannel3' : 256}):
    
        with tf.variable_scope(name,tf.AUTO_REUSE):
            # Placeholders are defined
            self.input = tf.placeholder(tf.float32, [None, input_dim[0],input_dim[1],input_dim[2]], name='Input_Image')
            self.true_labels = tf.placeholder(tf.float32, [None, output_dim], name='true_labelsification')
            self.is_training = tf.placeholder(tf.bool, name='is_training')    

            input_image = self.input
            true_labels = self.true_labels
            # essential line in case I use newer crossentropy function wich might allow gradient descent on labels as well
            tf.stop_gradient(true_labels,name='stop_GD_on_labels')

            ############ Layers creation   ###############
            
            '''EYE1 LAYERS'''
            eye1_output ,eye1_maxpoolLastConv = CreateEyeLayers(self, 
                                                                input_image, 
                                                                output_dim,
                                                                scope='eye1', 
                                                                optimizer=tf.train.AdamOptimizer, 
                                                                lr=0.01, 
                                                                filter_sizes = {'firstFilter' : 5, 'commonFilter': 3}, 
                                                                num_filters={'outChannel1' : 64, 'outChannel2': 128, 'outChannel3' : 256},
                                                                init_dict=eyeInitDictStage1()(),
                                                                trainable=True)

            '''DESCREMINATOR LAYERS'''
            desc_output = CreateDescreminatorLayers(self, 
                                                    eye1_maxpoolLastConv, 
                                                    output_dim=1,
                                                    isFalseEye=False,
                                                    scope='descreminator', 
                                                    optimizer=tf.train.AdamOptimizer, 
                                                    lr=0.01, 
                                                    filter_sizes = {'firstFilter' : 5, 'commonFilter': 3}, 
                                                    num_filters={'outChannel1' : 64, 'outChannel2': 128, 'outChannel3' : 256})
            
            # loss calculating part for true eye
            TrueConst = tf.get_variable(name="desc/TrueConst",shape=[1],initializer=tf.constant_initializer([1]),trainable=False)
            self.loss_desc_on_eye1 =tf.reduce_mean(tf.abs(tf.add(TrueConst, -desc_output,name='desc/TensoredTrueEyeDistanceFromOne'),
                                                   name='desc/trueEyeDistanceFromOne'),
                                                   name='desc/loss_desc_on_True_eye')
            
            #crossentropy due to 7 possible classifications 
            self.loss_eye1 = tf.nn.softmax_cross_entropy_with_logits_v2(
                                                    logits=eye1_output,labels=true_labels,name='eye1/cross_entropy')

            self.loss_eye1 = tf.reduce_mean(self.loss_eye1,name='eye1/loss')

            #summation of both descriminator classify and eye1 loss - this summation connects descreminator and eye1 calc paths 
            #in the flow graph - therefor applying optimizer on total_loss_desc_eye1 will affect  both paths
            self.total_loss_desc_eye1 =  tf.add(self.loss_eye1,self.loss_desc_on_eye1,name="total/eye1_plus_desc_loss")
            
            # When using the batchnormalization layers,
            # it is necessary to manually add the update operations
            # because the moving averages are not included in the graph            
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=name)
            with tf.control_dependencies(update_ops):                     
                self.train_op = optimizer(lr).minimize(self.total_loss_desc_eye1,name='eye1/Minimize')


            ''' ACCUARCY CALCULATIONS '''
            softmax_eye1_output = tf.nn.softmax(eye1_output, name='eye1/softmaxxed_eye1_output')
            
            # assume our eye gave the largest probability to class 5 - if the true class is indeed 5 -> correct_prediction = True
            self.accuracy_eye1 = tf.equal(tf.argmax(true_labels, axis=1,name='eye1/accuracy_eye1/argmaxTrueLabels'), 
                                          tf.argmax(softmax_eye1_output, axis=1,name='eye1/accuracy_eye1/argmaxPredictedLabels'),
                                          name='eye1/tensor_accuracy_eye1')
            
            #convert bool --> float32
            self.accuracy_eye1 = tf.reduce_mean(tf.cast(self.accuracy_eye1, tf.float32,name='eye1/caseAccuracyToFloat'),name='eye1/accuracy_eye1')
    


class Model_EWG_stage2:
 def __init__(self, 
                name, 
                input_dim, 
                output_dim,
                activeGraph=None,
                optimizer_TrueEyeAndDesc=tf.train.AdamOptimizer, 
                optimizer_FalseEyeAndDesc=tf.train.AdamOptimizer, 
                lr=0.01, filter_sizes = {'firstFilter' : 5, 'commonFilter': 3}, 
                num_filters={'outChannel1' : 64, 'outChannel2': 128, 'outChannel3' : 256, 'outChannel4' : 512}):

        # determines the beginning of each node's name in this model (e.g stage1/..)
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE): 
            # Placeholders are defined
            self.input = tf.placeholder(tf.float32, [None, input_dim[0],input_dim[1],input_dim[2]], name='Input_Image')
            self.true_labels = tf.placeholder(tf.float32, [None, output_dim], name='true_labelsification')
            self.is_training = tf.placeholder(tf.bool, name='is_training')    

            input_image = self.input
            true_labels = self.true_labels

            # essential line in case I use newer crossentropy function wich might allow gradient descent on labels as well
            tf.stop_gradient(true_labels,name='stop_GD_on_labels')

            ############ Layers creation   ###############

            '''EYE1 LAYERS'''
            '''weights are propagated from Model1'''
            eye1_output ,eye1_maxpoolLastConv = CreateEyeLayers(self, 
                                                                input_image, 
                                                                output_dim,
                                                                scope='eye1', 
                                                                optimizer=tf.train.AdamOptimizer, 
                                                                lr=0.01, 
                                                                filter_sizes = {'firstFilter' : 5, 'commonFilter': 3}, 
                                                                num_filters={'outChannel1' : 64, 'outChannel2': 128, 'outChannel3' : 256},
                                                                init_dict=eyeInitDictStage2(activeGraph=activeGraph)())
            
            '''EYE2 LAYERS'''
            eye2_output ,eye2_maxpoolLastConv = CreateEyeLayers(self, 
                                                                input_image, 
                                                                output_dim,
                                                                init_dict=eyeInitDictStage1()(),
                                                                scope='eye2', 
                                                                optimizer=tf.train.AdamOptimizer, 
                                                                lr=0.01, 
                                                                filter_sizes = {'firstFilter' : 5, 'commonFilter': 3}, 
                                                                num_filters={'outChannel1' : 64, 'outChannel2': 128, 'outChannel3' : 256})

            
            '''DESCREMINATOR LAYERS'''
            ''' 
            though it looks like we have just created 2 new descreminators - this is not the case.
            we actually created one descreminator which is initialized with the weights from Model1's descreminator training.
            but we created 2 defferent flows- for eye1 and eye2 ,because the descreminator affects each eye differently.
            for implementing such a functionality - we used the same scope namings - therefor, tensorflow treats it as the same.
            '''

            desc_output2 = CreateDescreminatorLayers(self, 
                                                    eye2_maxpoolLastConv, 
                                                    output_dim=1,
                                                    isFalseEye=True,
                                                    scope='descreminator', 
                                                    optimizer=tf.train.AdamOptimizer, 
                                                    lr=0.01, 
                                                    filter_sizes = {'firstFilter' : 5, 'commonFilter': 3}, 
                                                    num_filters={'outChannel1' : 64, 'outChannel2': 128, 'outChannel3' : 256},
                                                    init_dict=DescInitDictStage2(activeGraph=activeGraph)())
            
            desc_output1 = CreateDescreminatorLayers(self, 
                                                    eye1_maxpoolLastConv, 
                                                    output_dim=1,
                                                    isFalseEye=False,
                                                    scope='descreminator', 
                                                    optimizer=tf.train.AdamOptimizer, 
                                                    lr=0.01, 
                                                    filter_sizes = {'firstFilter' : 5, 'commonFilter': 3}, 
                                                    num_filters={'outChannel1' : 64, 'outChannel2': 128, 'outChannel3' : 256},
                                                    init_dict=DescInitDictStage2(activeGraph=activeGraph)())
            
    
            '''##########      DESCREMINATOR DECISION ON EYE1 (TRUE EYE)      ##########'''

            TrueConst = tf.get_variable(name="desc/TrueConst",shape=[1],initializer=tf.constant_initializer([1]),trainable=False)
            self.loss_desc_on_eye1 =tf.reduce_mean(tf.abs(tf.add(TrueConst, -desc_output1,name='desc/TensoredTrueEyeDistanceFromOne'),
                                                          name='desc/trueEyeDistanceFromOne'),
                                                    name='desc/trueEyeDistanceFromOne')
            
            #crossentropy due to 7 possible classifications and 
            self.loss_eye1 = tf.nn.softmax_cross_entropy_with_logits_v2(
                                                    logits=eye1_output,labels=true_labels,name='eye1/cross_entropy')

            self.loss_eye1 = tf.reduce_mean(self.loss_eye1,name='eye1/batchMeanloss')

            #summation of both descriminator and eye1 losses - this summation connects descreminator and eye1 calc paths 
            #in the graph - therefor applying optimizer on total_loss_desc_eye1 will affect  both paths
            total_loss_desc_eye1 =  tf.add(self.loss_eye1,self.loss_desc_on_eye1,name="total/eye1_plus_desc_loss")
            
            # When using the batchnormalization layers,
            # it is necessary to manually add the update operations
            # because the moving averages are not included in the graph    
            to_train = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                 scope="%s/[eye1|desc]" % name)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="%s/[eye1|desc]" % name)
            with tf.control_dependencies(update_ops):                     
                self.train_opTrue = optimizer_TrueEyeAndDesc(lr,name='eye1/TrueOptimizer').minimize(total_loss_desc_eye1,var_list=to_train,name='eye1/TrueMinimize')


            ''' ACCUARCY CALCULATIONS '''            
            softmax_eye1_output = tf.nn.softmax(eye1_output, name='eye1/softmaxxed_eye1_output')
            self.accuracy_eye1 = tf.equal(tf.argmax(true_labels, axis=1,name='eye1/accuracy_eye1/argmaxTrueLabels'), 
                                          tf.argmax(softmax_eye1_output, axis=1,name='eye1/accuracy_eye1/argmaxPredictedLabels'),
                                          name='eye1/tensor_accuracy_eye1')
            #convert bool --> float32
            self.accuracy_eye1 = tf.reduce_mean(tf.cast(self.accuracy_eye1, tf.float32,name='eye1/caseAccuracyToFloat'),name='eye1/accuracy_eye1')

            '''##########      DESCREMINATOR DECISION ON EYE2 (FALSE EYE)      ##########'''
            
            self.loss_desc_on_eye2 =tf.reduce_mean(tf.abs(desc_output2,name='desc/falseEyeDistanceFromOne'),
                                                    name='desc/false_eye_loss')
            #crossentropy due to 7 possible classifications and 
            self.loss_eye2 = tf.nn.softmax_cross_entropy_with_logits_v2(
                                                    logits=eye2_output,labels=true_labels,name='eye2/cross_entropy')

            self.loss_eye2 = tf.reduce_mean(self.loss_eye2,name='eye2/batchMeanloss')

            #summation of both descriminator and eye1 losses - this summation connects descreminator and eye1 calc paths 
            #in the graph - therefor applying optimizer on total_loss_desc_eye1 will affect  both paths
            total_loss_desc_eye2 =  tf.add(self.loss_eye2,self.loss_desc_on_eye2,name="total/eye2_plus_desc_loss")
            

            # When using the batchnormalization layers,
            # it is necessary to manually add the update operations
            # because the moving averages are not included in the graph   
            # print(stage2LoadLayersDict(activeGraph)())   
            to_train = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                 scope="%s/[eye2|desc]" % name)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="%s/[eye2|desc]" % name)
            with tf.control_dependencies(update_ops):                     
                self.train_opFalse = optimizer_FalseEyeAndDesc(lr,name='eye2/FalseOptimizer').minimize(total_loss_desc_eye2,var_list=to_train,name='eye2/FalseMinimize')


            '''ACCUARCY CALCULATIONS '''
            softmax_eye2_output = tf.nn.softmax(eye2_output, name='eye2/softmaxxed_eye2_output')
            self.accuracy_eye2 = tf.equal(tf.argmax(true_labels, axis=1,name='eye2/accuracy_eye2/argmaxTrueLabels'), 
                                          tf.argmax(softmax_eye2_output, axis=1,name='eye2/accuracy_eye2/argmaxPredictedLabels'),
                                          name='eye2/tensor_accuracy_eye2')
            #convert bool --> float32
            self.accuracy_eye2 = tf.reduce_mean(tf.cast(self.accuracy_eye2, tf.float32,name='eye2/caseAccuracyToFloat'),name='eye2/accuracy_eye2')


class Model_EWG_stage3:
 def __init__(self, 
                name, 
                input_dim, 
                output_dim,
                activeGraph=None,
                optimizer_TrueEyeAndDesc=tf.train.AdamOptimizer, 
                optimizer_FalseEyeAndDesc=tf.train.AdamOptimizer, 
                lr=0.01, filter_sizes = {'firstFilter' : 5, 'commonFilter': 3}, 
                num_filters={'outChannel1' : 64, 'outChannel2': 128, 'outChannel3' : 256, 'outChannel4' : 512}):

        with tf.variable_scope(name):
            # Placeholders are defined
            self.input = tf.placeholder(tf.float32, [None, input_dim[0],input_dim[1],input_dim[2]], name='Input_Image')
            self.true_labels = tf.placeholder(tf.float32, [None, output_dim], name='true_labelsification')
            self.is_training = tf.placeholder(tf.bool, name='is_training')    
            # self.areEyesTrainable = tf.placeholder(tf.bool, name='EyesAreTrainable')    

            input_image = self.input
            true_labels = self.true_labels

            # essential line in case I use newer crossentropy function wich might allow gradient descent on labels as well
            tf.stop_gradient(true_labels,name='stop_GD_on_labels')

            ############ Layers creation   ###############
            '''EYE1 LAYERS'''
            '''
            weights are initialized from Model2 training of eye1.
            eye1_maxpoolLastConv is the output of eye1 on image input
            '''
            eye1_output ,eye1_maxpoolLastConv = CreateEyeLayers(self, 
                                                                input_image, 
                                                                output_dim,
                                                                scope='eye1', 
                                                                optimizer=tf.train.AdamOptimizer, 
                                                                lr=0.01, 
                                                                filter_sizes = {'firstFilter' : 5, 'commonFilter': 3}, 
                                                                num_filters={'outChannel1' : 64, 'outChannel2': 128, 'outChannel3' : 256},
                                                                init_dict=eye1InitDictStage3(activeGraph=activeGraph)(),
                                                                trainable=True)
            '''EYE2 LAYERS'''
            '''
            weights are initialized from Model2 training of eye2
            eye2_maxpoolLastConv is the output of eye1 on image input
            '''
            eye2_output ,eye2_maxpoolLastConv = CreateEyeLayers(self, 
                                                                input_image, 
                                                                output_dim,
                                                                scope='eye2', 
                                                                optimizer=tf.train.AdamOptimizer, 
                                                                lr=0.01, 
                                                                filter_sizes = {'firstFilter' : 5, 'commonFilter': 3}, 
                                                                num_filters={'outChannel1' : 64, 'outChannel2': 128, 'outChannel3' : 256},
                                                                init_dict=eye2InitDictStage3(activeGraph=activeGraph)(),
                                                                trainable=True)

            ''' BRAIN LAYERS '''
            brain_output = CreateBrainLayers(self, 
                                        eye1_maxpoolLastConv, 
                                        eye2_maxpoolLastConv,
                                        output_dim=output_dim,
                                        scope='brain', 
                                        optimizer=tf.train.AdamOptimizer, 
                                        lr=0.01, 
                                        filter_sizes = {'firstFilter' : 5, 'commonFilter': 3}, 
                                        num_filters={'outChannel1' : 64, 'outChannel2': 128, 'outChannel3' : 256},
                                        init_dict=BrainInitDictStage1()())
            
            '''##########      LOSSES EYES AND BRAIN     ##########'''
            self.loss_brain = SpicyLoss(  self,
                                        brain_output,
                                        true_labels,
                                        scope='brain',
                                        name='cross_entropy')
            
            self.loss_eye1 = SpicyLoss(  self,
                                        eye1_output,
                                        true_labels,
                                        scope='eye1',
                                        name='cross_entropy')
            self.loss_eye2 = SpicyLoss(  self,
                                        eye2_output,
                                        true_labels,
                                        scope='eye1',
                                        name='cross_entropy')

            self.weighted_predictions= brain_weight*brain_output + eye1_weight*eye1_output + eye2_weight*eye2_output


            self.loss_weighted = SpicyLoss(  self,
                                        self.weighted_predictions,
                                        true_labels,
                                        scope='weighted',
                                        name='cross_entropy')

            '''##########      BRAIN OPTIMIZER ON EYES     ##########'''
            #get all trainable variables and intersect them only 
            # with the variables that are from stage3 and related to eye1
            to_train1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                 scope="%s/eye1/" % name)

            # When using the batchnormalization layers,
            # it is necessary to manually add the update operations
            # because the moving averages are not included in the graph   
            update_ops1 = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="%s/eye1" % name)
            with tf.control_dependencies(update_ops1):  
                self.train_opTrue = optimizer_FalseEyeAndDesc(lr,name='eye1/Optimizer').minimize(self.loss_eye1,
                                                                                                 var_list=to_train1,                #     
                                                                                                  name='eye1/Minimize')


            # get all trainable variables and intersect them only with the 
            # variables that are from stage3 and related to eye2
            to_train2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                 scope="%s/eye2/" % name)
    
            update_ops2 = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="%s/eye2" % name)
            with tf.control_dependencies(update_ops2):                     
                self.train_opFalse = optimizer_FalseEyeAndDesc(lr,name='eye2/Optimizer').minimize(self.loss_eye2,
                                                                                            var_list=to_train2,
                                                                                                    name='eye2/Minimize')

            # get all trainable variables and intersect them only with the 
            # variables that are from stage3 and related to brain
            to_train2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                 scope="%s/brain/" % name)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="%s/brain" % name)
            with tf.control_dependencies(update_ops):                     
                self.train_opBrain = optimizer_FalseEyeAndDesc(lr,name='brain/Optimizer').minimize(self.loss_brain,
                                                                                                    name='brain/OptimizeMinimize')

            ''' ACCURACIES OF EYES AND BRAIN '''
            
            self.accuracy_brain = SpicyAccuracy(self,
                                        predictions=brain_output,
                                        labels=true_labels,
                                        scope='brain',
                                        name='softmaxxed_brain_output')

            self.accuracy_eye1 = SpicyAccuracy(self,
                                        predictions=eye1_output,
                                        labels=true_labels,
                                        scope='eye1',
                                        name='softmaxxed_brain_output')
            self.accuracy_eye2 = SpicyAccuracy(self,
                                        predictions=eye2_output,
                                        labels=true_labels,
                                        scope='eye2',
                                        name='softmaxxed_brain_output')
            self.accuracy_weighted = SpicyAccuracy(self,
                                        predictions=self.weighted_predictions,
                                        labels=true_labels,
                                        scope='weighted',
                                        name='softmaxxed_brain_output')
           
           


class Solver:
    """
    Solver class
    is created to train and evaluate each Model[123] instance from the models above. Thus, it
    has methods for training and evaluating each stage of training
    """
    def __init__(self, sess, model):
        self.model = model
        self.sess = sess
        
    def trainStage1(self, input_images, true_labels):
        feed = {
            self.model.input: input_images,
            self.model.true_labels: true_labels,
            self.model.is_training:   True
        }

        train_op = self.model.train_op 
        loss_eye1 = self.model.loss_eye1
        loss_desc_on_eye1 = self.model.loss_desc_on_eye1

        return self.sess.run([train_op, loss_eye1,loss_desc_on_eye1], feed_dict=feed)

    def trainStage2(self, input_images, true_labels, isTrueEye):
        feed = {
            self.model.input: input_images,
            self.model.true_labels: true_labels,
            self.model.is_training:   True
        }

        trainOpTrueEye = self.model.train_opTrue 
        trainOpFalseEye = self.model.train_opFalse
        
        loss_eye1 = self.model.loss_eye1
        loss_eye2 = self.model.loss_eye2

        loss_desc_on_eye1 = self.model.loss_desc_on_eye1
        loss_desc_on_eye2 = self.model.loss_desc_on_eye2

        if True == isTrueEye:
            return self.sess.run([trainOpTrueEye, loss_eye1,loss_desc_on_eye1], feed_dict=feed)
        else:
            return self.sess.run([trainOpFalseEye, loss_eye2,loss_desc_on_eye2], feed_dict=feed)

    def trainStage3(self, input_images, true_labels,areEyesTrainable=True):
        feed = {
            self.model.input: input_images,
            self.model.true_labels: true_labels,
            self.model.is_training:   True
        }
        train_opBrain = self.model.train_opBrain 
        train_opFalse = self.model.train_opFalse
        train_opTrue = self.model.train_opTrue
        loss_brain = self.model.loss_brain

        #if eyes are trainable- optimize them too
        if areEyesTrainable:
            self.sess.run([train_opFalse], feed_dict=feed)
            self.sess.run([train_opTrue],feed_dict=feed)
        return self.sess.run([train_opBrain,loss_brain], feed_dict=feed)


    
    def evaluateStage1(self, checkDataSet,checkDataLabels , batch_size=512):
        N = checkDataSet.shape[0]
        
        total_loss_eye1 = 0
        total_loss_desc_on_eye1 = 0
        total_acc_eye1 = 0
        numOfSteps = 0
        for i in range(0, N, batch_size):
            numOfSteps+=1
            X_batch = checkDataSet[i:i + batch_size]
            y_batch = checkDataLabels[i:i + batch_size]
            
            feed = {
                self.model.input: X_batch,
                self.model.true_labels: y_batch,
                self.model.is_training: False
            }
            
            loss_eye1 = self.model.loss_eye1
            loss_desc_on_eye1 = self.model.loss_desc_on_eye1
            accuracy_eye1 = self.model.accuracy_eye1
            step_loss_eye1, step_acc_eye1, step_loss_desc_on_eye1  =  \
            self.sess.run([loss_eye1, accuracy_eye1,loss_desc_on_eye1], feed_dict=feed)
            
            total_loss_eye1 += step_loss_eye1
            total_acc_eye1 += step_acc_eye1
            total_loss_desc_on_eye1 += step_loss_desc_on_eye1
        total_loss_eye1 /= numOfSteps
        total_loss_desc_on_eye1 /= numOfSteps
        total_acc_eye1 /= numOfSteps
        
        return total_loss_eye1, total_acc_eye1, total_loss_desc_on_eye1

    def evaluateStage2(self, checkDataSet,checkDataLabels , batch_size=512):
    
        N = checkDataSet.shape[0]
        
        total_loss_eye1 = 0
        total_loss_desc_on_eye1 = 0
        total_acc_eye1 = 0
        total_loss_eye2 = 0
        total_loss_desc_on_eye2 = 0
        total_acc_eye2 = 0
        numOfSteps=0
        for i in range(0, N, batch_size):
            numOfSteps+=1
            X_batch = checkDataSet[i:i + batch_size]
            y_batch = checkDataLabels[i:i + batch_size]
            
            feed = {
                self.model.input: X_batch,
                self.model.true_labels: y_batch,
                self.model.is_training: False
            }
            
            loss_eye1 = self.model.loss_eye1
            loss_desc_on_eye1 = self.model.loss_desc_on_eye1
            accuracy_eye1 = self.model.accuracy_eye1
            loss_eye2 = self.model.loss_eye2
            loss_desc_on_eye2 = self.model.loss_desc_on_eye2
            accuracy_eye2 = self.model.accuracy_eye2
            
            step_loss_eye1, step_acc_eye1, step_loss_desc_on_eye1,step_loss_eye2, step_acc_eye2, step_loss_desc_on_eye2  =  \
            self.sess.run([loss_eye1, accuracy_eye1,loss_desc_on_eye1,loss_eye2, accuracy_eye2,loss_desc_on_eye2], feed_dict=feed)
            
            total_loss_eye1 += step_loss_eye1
            total_acc_eye1 += step_acc_eye1
            total_loss_desc_on_eye1 += step_loss_desc_on_eye1
            total_loss_eye2 += step_loss_eye2
            total_acc_eye2 += step_acc_eye2
            total_loss_desc_on_eye2 += step_loss_desc_on_eye2
        total_loss_eye1 /= numOfSteps
        total_loss_desc_on_eye1 /= numOfSteps
        total_acc_eye1 /= numOfSteps
        total_loss_eye2 /= numOfSteps
        total_loss_desc_on_eye2 /= numOfSteps
        total_acc_eye2 /= numOfSteps
        
        return total_loss_eye1, total_acc_eye1, total_loss_desc_on_eye1,total_loss_eye2, total_acc_eye2, total_loss_desc_on_eye2

    def evaluateStage3(self, checkDataSet,checkDataLabels , batch_size=512):
    
        N = checkDataSet.shape[0]
        
        avg_loss_brain = 0
        avg_acc_brain = 0
        avg_loss_eye1=0
        avg_acc_eye1=0
        avg_loss_eye2=0
        avg_acc_eye2=0
        avg_loss_weighted = 0
        avg_acc_weighted = 0
        numOfSteps=0
        for i in range(0, N, batch_size):
            numOfSteps+=1
            X_batch = checkDataSet[i:i + batch_size]
            y_batch = checkDataLabels[i:i + batch_size]
            
            feed = {
                self.model.input: X_batch,
                self.model.true_labels: y_batch,
                self.model.is_training: False
            }
            
            loss_brain = self.model.loss_brain
            accuracy_brain = self.model.accuracy_brain

            loss_eye1 = self.model.loss_eye1
            loss_eye2 = self.model.loss_eye2
            loss_weighted = self.model.loss_weighted
            
            accuracy_eye1 = self.model.accuracy_eye1
            accuracy_eye2 = self.model.accuracy_eye2
            accuracy_weighted = self.model.accuracy_weighted

            step_loss_brain, step_acc_brain,step_loss_eye1,step_acc_eye1,step_loss_eye2,step_acc_eye2 ,step_loss_weighted,step_accuracy_weighted =  \
            self.sess.run([loss_brain, accuracy_brain,loss_eye1,accuracy_eye1,loss_eye2,accuracy_eye2,loss_weighted,accuracy_weighted], feed_dict=feed)

            avg_loss_brain += step_loss_brain
            avg_acc_brain += step_acc_brain
            avg_loss_eye1 += step_loss_eye1
            avg_acc_eye1 += step_acc_eye1
            avg_loss_eye2 += step_loss_eye2
            avg_acc_eye2 += step_acc_eye2
            avg_loss_weighted += step_loss_weighted
            avg_acc_weighted += step_accuracy_weighted

        avg_loss_brain /= numOfSteps
        avg_acc_brain /= numOfSteps
        avg_loss_eye1 /= numOfSteps
        avg_acc_eye1  /= numOfSteps
        avg_loss_eye2 /= numOfSteps
        avg_acc_eye2  /= numOfSteps
        avg_loss_weighted /= numOfSteps
        avg_acc_weighted /= numOfSteps
        return avg_loss_brain, avg_acc_brain,avg_loss_eye1,avg_acc_eye1,avg_loss_eye2,avg_acc_eye2,avg_loss_weighted,avg_acc_weighted
        

