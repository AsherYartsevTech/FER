from GlobalVarsAndLibs import *

def runSessStage1(execSess,bn_solver,train_dataset,train_labels,valid_dataset,valid_labels,test_dataset,test_labels,fastRun=False):
    '''
    FUNCTION MUST BE EXECUTED UNDER OPEN GRAPH SESSION
    '''
    
    epoch_n = np.floor(epochsNum/3)
    batch_size = 512
    N = 172254
    init = tf.global_variables_initializer()
    ########### SESSION RUNNING ##########
    #for saving the weiths we trained
    saver = tf.train.Saver()
    execSess.run(init)

    print('\n\n ########## [STAGE1] TRAINING MODE  ####### \n\n')
    for epoch in range(epoch_n):
        for step in range(N//batch_size):
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
            batch_labels = train_labels[offset:(offset + batch_size), :]        
            
            _, loss_eye1 , loss_desc_on_eye1 = bn_solver.trainStage1(batch_data, batch_labels) 
        avg_loss_eye1, avg_acc_eye1, avg_loss_desc_on_eye1 = \
        bn_solver.evaluateStage1(valid_dataset,valid_labels)

        print(f'[Epoch {epoch}-TRAIN] ||  eye1 loss(Acc)<validation set>: {avg_loss_eye1:.5f}({avg_acc_eye1:.2%}) \
        || loss_desc_on_eye1 loss<validation set>:{avg_loss_desc_on_eye1:.5f}')
        print(f'loss_eye1<training set>: {loss_eye1:.5f} \
        || loss_desc_on_eye1<training set>:{loss_desc_on_eye1:.5f}')
    save_path = saver.save(execSess, "nets/eyesWithGunsModule/saved_weights/Stage1Model_final_weights.ckpt")
    

def runSessStage2(execSess,bn_solver,train_dataset,train_labels,valid_dataset,valid_labels,test_dataset,test_labels,fastRun=False):
    '''
    FUNCTION MUST BE EXECUTED UNDER OPEN GRAPH SESSION
    '''
    epoch_n = epochsNum
    batch_size = 512
    N = 172254

    '''########### SESSION RUNNING ##########'''
    saver = tf.train.Saver()      # for saving the weiths we trained
    init = tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                 scope="stage2/"))
    execSess.run(init)
    print('\n\n ########## [STAGE2] TRAINING MODE  ####### \n\n')
    for epoch in range(epoch_n):
        for step in range(N//batch_size):
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
            batch_labels = train_labels[offset:(offset + batch_size), :]       
            # modolu parameter to trainStage2 decides which eye to train 
            _, loss_eye , loss_desc_on_eye = bn_solver.trainStage2(batch_data, batch_labels, (step%2)) 
        avg_loss_eye1, avg_acc_eye1, avg_loss_desc_on_eye1,avg_loss_eye2, avg_acc_eye2, avg_loss_desc_on_eye2 = \
        bn_solver.evaluateStage2(valid_dataset,valid_labels)
        print(f'[Epoch {epoch}-TRAIN] ||  eye1 loss(Acc): {avg_loss_eye1:.5f}({avg_acc_eye1:.2%}) \
        || loss_desc_on_eye1 loss:{avg_loss_desc_on_eye1:.5f}')
        print(f'[Epoch {epoch}-TRAIN] ||  eye2 loss(Acc): {avg_loss_eye2:.5f}({avg_acc_eye2:.2%}) \
        || loss_desc_on_eye2 loss:{avg_loss_desc_on_eye2:.5f}')
    save_path = saver.save(execSess, "nets/eyesWithGunsModule/saved_weights/Stage2Model_final_weights.ckpt")

def runSessStage3(execSess,bn_solver,train_dataset,train_labels,valid_dataset,valid_labels,test_dataset,test_labels,fastRun=False):
    '''
    FUNCTION MUST BE EXECUTED UNDER OPEN GRAPH SESSION
    '''
    epoch_n = epochsNum
    batch_size = 512
    N = 172254
    ########### SESSION RUNNING ##########
    #for saviing the weiths we trained
    saver = tf.train.Saver()
    init = tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                 scope="stage3/"))
    execSess.run(init)
    print('\n\n ########## [STAGE3] TRAINING MODE  ####### \n\n')
    for epoch in range(epoch_n):
        for step in range(N//batch_size):
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
            batch_labels = train_labels[offset:(offset + batch_size), :]   
            _, loss_brain  = bn_solver.trainStage3(batch_data, batch_labels,areEyesTrainable=True)
        avg_loss_brain, avg_acc_brain,avg_loss_eye1,avg_acc_eye1,avg_loss_eye2,avg_acc_eye2,avg_weighted_loss,avg_weighted_acc = \
        bn_solver.evaluateStage3(valid_dataset,valid_labels)

        print(f'[Epoch {epoch}-TRAIN] STATS:\nbrain loss(Acc): {avg_loss_brain:.5f}({avg_acc_brain:.2%}), \
        eye1 loss(Acc): {avg_loss_eye1:.5f}({avg_acc_eye1:.2%}), \
        eye2 loss(Acc): {avg_loss_eye2:.5f}({avg_acc_eye2:.2%})  \
        weighted loss(Acc): {avg_weighted_loss:.5f}({avg_weighted_acc:.2%})')
        avg_loss_brain, avg_acc_brain,avg_loss_eye1,avg_acc_eye1,avg_loss_eye2,avg_acc_eye2,avg_weighted_loss,avg_weighted_acc = \
        bn_solver.evaluateStage3(test_dataset,test_labels)

        print(f'[Epoch {epoch}-TEST] || brain loss(Acc):{avg_loss_brain:.5f}({avg_acc_brain:.2%})')
        print(f'[Epoch {epoch}-TEST] || eye1 loss(Acc): {avg_loss_eye1:.5f}( {avg_acc_eye1:.2%})')
        print(f'[Epoch {epoch}-TEST] || eye2 loss(Acc): {avg_loss_eye2:.5f}( {avg_acc_eye2:.2%})')
        print(f'[Epoch {epoch}-TEST] || weighted loss(Acc): {avg_weighted_loss:.5f}( {avg_weighted_acc:.2%})')
    save_path = saver.save(execSess, "nets/eyesWithGunsModule/saved_weights/Stage3Model_final_weights.ckpt")
    