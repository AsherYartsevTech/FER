from datetime import datetime
from GlobalVarsAndLibs import *
from ImageInputMaker import getNetDataInput
from Models import *
from runSessions import *
import os as os
import subprocess
from subprocess import call

#print source control version
retval = os.getcwd()
os.chdir(retval+"/eyesWithGans")
pwd=call("pwd")
print(pwd)
label = subprocess.check_output(["git", "rev-parse",'--short','HEAD']).strip()
print('get version:',label)
os.chdir(retval)

''' create folder to store logs for TensorBoard analyzing '''
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "nets/eyesWithGunsModule/tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)


'''acquire data for net'''
train_dataset, train_labels,valid_dataset, valid_labels,test_dataset, test_labels =  getNetDataInput()


SpicyGraph = tf.Graph()
with SpicyGraph.as_default():
    
    '''######## STAGE 1 #############'''
    '''train eye1 and descreminator on eye1 to catch a training gap on eye2 - therefor ensuring both eyes won't 
        converge to similar distributions '''
    model = Model_EWG_stage1(name='stage1', input_dim=[image_size,image_size,input_n_channels], output_dim=n_classes)


    execSess = tf.Session()
    with execSess.as_default():

        bn_solver = Solver(execSess, model)
        runSessStage1(execSess,bn_solver,
                    train_dataset,train_labels,
                    valid_dataset,valid_labels,
                    test_dataset,test_labels)
        

        '''######## STAGE 2 #############'''
        '''train both eyes with descreminator - eye1 is considered as "True Data" and eye2 is considered as "false data". 
        both eyes generate weights according to same input data [images]. yet, arbitrary labling of true and false eye, helps the descriminator
        to add values closer to 1 to eye1's loss , and values closer to 0 to eye2 loss - thus, encouraging eyes to converge to different
        distributions - which is analog to focusing on different features - which theoretically might produce much better results
        when applying convolutional networks '''
        model = Model_EWG_stage2(name='stage2',activeGraph=SpicyGraph, input_dim=[image_size,image_size,input_n_channels], output_dim=n_classes)
    
        bn_solver = Solver(execSess, model)
        # if True == debug:
        #     debugWeightTransformSt1toSt2(SpicyGraph)
        runSessStage2(execSess,bn_solver,
                    train_dataset,train_labels,
                    valid_dataset,valid_labels,
                    test_dataset,test_labels)        

        '''######## STAGE 3 #############'''
        ''' in this stage both eyes developed unique feature maps,so it makes sense to add a "brain component" 
        which is simply few fully connected layers - which is trained on eyes features , instead on the input images.
        Finally, when a test set is provided, we are averaging the classification opinions of the brain, eye1 and eye2,
        whith slight preferability for brain classification'''
        model = Model_EWG_stage3(name='stage3',activeGraph=SpicyGraph, input_dim=[image_size,image_size,input_n_channels], output_dim=n_classes)

        bn_solver = Solver(execSess, model)
        runSessStage3(execSess,bn_solver,
                    train_dataset,train_labels,
                    valid_dataset,valid_labels,
                    test_dataset,test_labels)
file_writer = tf.summary.FileWriter(logdir, SpicyGraph)




            

