from GlobalVarsAndLibs import *
import sys

def SpicyLoss(  self,
                predictions,
                labels,
                scope='noScope',
                name='loss_noName'):
                loss=tf.nn.softmax_cross_entropy_with_logits_v2( logits=predictions,
                                                                labels=labels,
                                                                name='%s/%s' %(scope,name))
                loss=tf.reduce_mean(loss,name='%s/%s/loss' %(scope,name))
                return loss

def SpicyAccuracy(self,
                  predictions,
                  labels,
                  scope='noScope',
                  name='Accuracy_noName'):
    '''
    predictions - logits, a product of a matmul without any activation function interferance
    labels - true labels
    '''
    softmax_output=tf.nn.softmax(predictions, name='%s/%s' % (scope,name))
    accuracy = tf.equal(tf.argmax(labels, axis=1,name='%s/%s/accuracy_brain/argmaxTrueLabels' % (scope,name)), 
                                tf.argmax(softmax_output, axis=1,name='%s/%s/accuracy_brain/argmaxPredictedLabels' % (scope,name)),
                                name='%s/%s/tensor_accuracy_brain'% (scope,name))
    accuracy = tf.reduce_mean(tf.cast(accuracy, 
                                        tf.float32,
                                        name='%s/%s/caseAccuracyToFloat'% (scope,name)),
                                name='%s/%s/accuracy_brain'% (scope,name))
    return accuracy
