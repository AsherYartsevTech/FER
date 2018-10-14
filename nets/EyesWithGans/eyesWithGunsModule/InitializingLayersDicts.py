from GlobalVarsAndLibs import *

class eyeInitDictStage1:
    def __init__(self,
                 weightsInit=tf.truncated_normal_initializer(stddev=0.04),
                 biasInit= tf.zeros_initializer(),gammaInit=None,betaInit=None):

        self.Dict = {
'conv1/weights'       :   weightsInit,
'conv1/biases'        :   biasInit,
'batchNorm1/gamma'    :   gammaInit,
'batchNorm1/beta'     :   betaInit,

'conv2/weights'       :   weightsInit,
'conv2/biases'        :   biasInit,
'batchNorm2/gamma'        :   gammaInit,
'batchNorm2/beta'     :   betaInit,

'conv3/weights'       :   weightsInit,
'conv3/biases'        :   biasInit,
'batchNorm3/gamma'        :   gammaInit,
'batchNorm3/beta'     :   betaInit,

'conv4/weights'       :   weightsInit,
'conv4/biases'        :   biasInit,
'batchNorm4/gamma'        :   gammaInit,
'batchNorm4/beta'     :   betaInit,

'conv5/weights'       :   weightsInit,
'conv5/biases'        :   biasInit,
'batchNorm5/gamma'        :   gammaInit,
'batchNorm5/beta'     :   betaInit,

'conv6/weights'       :   weightsInit,
'conv6/biases'        :   biasInit,
'batchNorm6/gamma'        :   gammaInit,
'batchNorm6/beta'     :   betaInit,

'conv7/weights'       :   weightsInit,
'conv7/biases'        :   biasInit,
'batchNorm7/gamma'        :   gammaInit,
'batchNorm7/beta'     :   betaInit,

'conv8/weights'       :   weightsInit,
'conv8/biases'        :   biasInit,
'batchNorm8/gamma'        :   gammaInit,
'batchNorm8/beta'     :   betaInit,

'conv9/weights'       :   weightsInit,
'conv9/biases'        :   biasInit,
'batchNorm9/gamma'        :   gammaInit,
'batchNorm9/beta'     :   betaInit,

'conv10/weights'      :   weightsInit,
'conv10/biases'       :   biasInit,
'batchNorm10/gamma'       :   gammaInit,
'batchNorm10/beta'        :   betaInit,

'dense1/weights'      :   weightsInit,
'dense1/biases'       :   biasInit,
'batchNorm11/gamma'       :   gammaInit,
'batchNorm11/beta'        :   betaInit,

'dense2/weights'      :   weightsInit,
'dense2/biases'       :   biasInit,
'batchNorm12/gamma'       :   gammaInit,
'batchNorm12/beta'        :   betaInit,

'classify/weights'        :   weightsInit,
'classify/biases'     :   biasInit
        }


    def __call__(self):
        return self.Dict


class DescInitDictStage1:
    def __init__(self,
                 weightsInit=tf.truncated_normal_initializer(stddev=0.04),
                 biasInit= tf.zeros_initializer(),gammaInit=None,betaInit=None):
        self.Dict = {
'conv1/weights'      :   weightsInit,
'conv1/biases'       :   biasInit,
'batchNorm1/gamma'       :   gammaInit,
'batchNorm1/beta'        :   betaInit,

'conv2/weights'      :   weightsInit,
'conv2/biases'       :   biasInit,
'batchNorm2/gamma'       :   gammaInit,
'batchNorm2/beta'        :   betaInit,

'dense1/weights'     :   weightsInit,
'dense1/biases'      :   biasInit,
'batchNorm3/gamma'       :   gammaInit,
'batchNorm3/beta'        :   betaInit,

'dense2/weights'     :   weightsInit,
'dense2/biases'      :   biasInit,
'batchNorm4/gamma'       :   gammaInit,
'batchNorm4/beta'        :   betaInit,
'TrueProbability/weights'        :   weightsInit,
'TrueProbability/biases'     :   biasInit,
'FalseProbability/weights'        :   weightsInit,
'FalseProbability/biases'     :   biasInit
                    }
    def __call__(self):
        return self.Dict




class eyeInitDictStage2:
    def __init__(self,
                 activeGraph,
                 weightsInit=tf.truncated_normal_initializer(stddev=0.04),
                 biasInit= tf.zeros_initializer(),gammaInit=None,betaInit=None):

        self.Dict = {
'conv1/weights'             :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage1/eye1/conv1/weights:0').eval()),
'conv1/biases'              :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage1/eye1/conv1/biases:0').eval()),
'batchNorm1/gamma'          :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage1/eye1/batchNorm1/gamma:0').eval()),
'batchNorm1/beta'           :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage1/eye1/batchNorm1/beta:0').eval()),
'conv2/weights'             :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage1/eye1/conv2/weights:0').eval()),
'conv2/biases'              :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage1/eye1/conv2/biases:0').eval()),
'batchNorm2/gamma'          :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage1/eye1/batchNorm2/gamma:0').eval()),
'batchNorm2/beta'           :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage1/eye1/batchNorm2/beta:0').eval()),
'conv3/weights'             :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage1/eye1/conv3/weights:0').eval()),
'conv3/biases'              :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage1/eye1/conv3/biases:0').eval()),
'batchNorm3/gamma'          :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage1/eye1/batchNorm3/gamma:0').eval()),
'batchNorm3/beta'           :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage1/eye1/batchNorm3/beta:0').eval()),
'conv4/weights'             :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage1/eye1/conv4/weights:0').eval()),
'conv4/biases'              :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage1/eye1/conv4/biases:0').eval()),
'batchNorm4/gamma'          :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage1/eye1/batchNorm4/gamma:0').eval()),
'batchNorm4/beta'           :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage1/eye1/batchNorm4/beta:0').eval()),
'conv5/weights'             :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage1/eye1/conv5/weights:0').eval()),
'conv5/biases'              :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage1/eye1/conv5/biases:0').eval()),
'batchNorm5/gamma'          :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage1/eye1/batchNorm5/gamma:0').eval()),
'batchNorm5/beta'           :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage1/eye1/batchNorm5/beta:0').eval()),
'conv6/weights'             :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage1/eye1/conv6/weights:0').eval()),
'conv6/biases'              :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage1/eye1/conv6/biases:0').eval()),
'batchNorm6/gamma'          :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage1/eye1/batchNorm6/gamma:0').eval()),
'batchNorm6/beta'           :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage1/eye1/batchNorm6/beta:0').eval()),
'conv7/weights'             :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage1/eye1/conv7/weights:0').eval()),
'conv7/biases'              :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage1/eye1/conv7/biases:0').eval()),
'batchNorm7/gamma'          :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage1/eye1/batchNorm7/gamma:0').eval()),
'batchNorm7/beta'           :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage1/eye1/batchNorm7/beta:0').eval()),
'conv8/weights'             :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage1/eye1/conv8/weights:0').eval()),
'conv8/biases'              :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage1/eye1/conv8/biases:0').eval()),
'batchNorm8/gamma'          :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage1/eye1/batchNorm8/gamma:0').eval()),
'batchNorm8/beta'           :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage1/eye1/batchNorm8/beta:0').eval()),
'conv9/weights'             :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage1/eye1/conv9/weights:0').eval()),
'conv9/biases'              :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage1/eye1/conv9/biases:0').eval()),
'batchNorm9/gamma'          :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage1/eye1/batchNorm9/gamma:0').eval()),
'batchNorm9/beta'           :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage1/eye1/batchNorm9/beta:0').eval()),
'conv10/weights'            :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage1/eye1/conv10/weights:0').eval()),
'conv10/biases'             :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage1/eye1/conv10/biases:0').eval()),
'batchNorm10/gamma'         :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage1/eye1/batchNorm10/gamma:0').eval()),
'batchNorm10/beta'          :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage1/eye1/batchNorm10/beta:0').eval()),
'dense1/weights'            :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage1/eye1/dense1/weights:0').eval()),
'dense1/biases'             :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage1/eye1/dense1/biases:0').eval()),
'batchNorm11/gamma'         :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage1/eye1/batchNorm11/gamma:0').eval()),
'batchNorm11/beta'          :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage1/eye1/batchNorm11/beta:0').eval()),
'dense2/weights'            :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage1/eye1/dense2/weights:0').eval()),
'dense2/biases'             :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage1/eye1/dense2/biases:0').eval()),
'batchNorm12/gamma'         :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage1/eye1/batchNorm12/gamma:0').eval()),
'batchNorm12/beta'          :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage1/eye1/batchNorm12/beta:0').eval()),
'classify/weights'          :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage1/eye1/classify/weights:0').eval()),
'classify/biases'           :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage1/eye1/classify/biases:0').eval())
        }


    def __call__(self):
        return self.Dict


class DescInitDictStage2:
    def __init__(self,
                 activeGraph,    
                 weightsInit=tf.truncated_normal_initializer(stddev=0.04),
                 biasInit= tf.zeros_initializer(),gammaInit=None,betaInit=None):
        self.Dict = {
'conv1/weights'             :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage1/descreminator/conv1/weights:0').eval()),
'conv1/biases'              :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage1/descreminator/conv1/biases:0').eval()),
'batchNorm1/gamma'          :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage1/descreminator/batchNorm1/gamma:0').eval()),
'batchNorm1/beta'           :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage1/descreminator/batchNorm1/beta:0').eval()),
'conv2/weights'             :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage1/descreminator/conv2/weights:0').eval()),
'conv2/biases'              :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage1/descreminator/conv2/biases:0').eval()),
'batchNorm2/gamma'          :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage1/descreminator/batchNorm2/gamma:0').eval()),
'batchNorm2/beta'           :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage1/descreminator/batchNorm2/beta:0').eval()),
'dense1/weights'            :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage1/descreminator/dense1/weights:0').eval()),
'dense1/biases'             :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage1/descreminator/dense1/biases:0').eval()),
'batchNorm3/gamma'          :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage1/descreminator/batchNorm3/gamma:0').eval()),
'batchNorm3/beta'           :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage1/descreminator/batchNorm3/beta:0').eval()),
'dense2/weights'            :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage1/descreminator/dense2/weights:0').eval()),
'dense2/biases'             :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage1/descreminator/dense2/biases:0').eval()),
'batchNorm4/gamma'          :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage1/descreminator/batchNorm4/gamma:0').eval()),
'batchNorm4/beta'           :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage1/descreminator/batchNorm4/beta:0').eval()),
'TrueProbability/weights'   :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage1/descreminator/TrueProbability/weights:0').eval()),
'TrueProbability/biases'    :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage1/descreminator/TrueProbability/biases:0').eval()),
'FalseProbability/weights'   :   weightsInit,
'FalseProbability/biases'    :   biasInit
    }
    def __call__(self):
        return self.Dict

class eye1InitDictStage3:
    def __init__(self,
                 activeGraph,
                 weightsInit=tf.truncated_normal_initializer(stddev=0.04),
                 biasInit= tf.zeros_initializer(),gammaInit=None,betaInit=None):

        self.Dict = {
'conv1/weights'             :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye1/conv1/weights:0').eval()),
'conv1/biases'              :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye1/conv1/biases:0').eval()),
'batchNorm1/gamma'          :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye1/batchNorm1/gamma:0').eval()),
'batchNorm1/beta'           :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye1/batchNorm1/beta:0').eval()),
'conv2/weights'             :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye1/conv2/weights:0').eval()),
'conv2/biases'              :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye1/conv2/biases:0').eval()),
'batchNorm2/gamma'          :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye1/batchNorm2/gamma:0').eval()),
'batchNorm2/beta'           :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye1/batchNorm2/beta:0').eval()),
'conv3/weights'             :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye1/conv3/weights:0').eval()),
'conv3/biases'              :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye1/conv3/biases:0').eval()),
'batchNorm3/gamma'          :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye1/batchNorm3/gamma:0').eval()),
'batchNorm3/beta'           :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye1/batchNorm3/beta:0').eval()),
'conv4/weights'             :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye1/conv4/weights:0').eval()),
'conv4/biases'              :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye1/conv4/biases:0').eval()),
'batchNorm4/gamma'          :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye1/batchNorm4/gamma:0').eval()),
'batchNorm4/beta'           :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye1/batchNorm4/beta:0').eval()),
'conv5/weights'             :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye1/conv5/weights:0').eval()),
'conv5/biases'              :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye1/conv5/biases:0').eval()),
'batchNorm5/gamma'          :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye1/batchNorm5/gamma:0').eval()),
'batchNorm5/beta'           :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye1/batchNorm5/beta:0').eval()),
'conv6/weights'             :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye1/conv6/weights:0').eval()),
'conv6/biases'              :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye1/conv6/biases:0').eval()),
'batchNorm6/gamma'          :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye1/batchNorm6/gamma:0').eval()),
'batchNorm6/beta'           :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye1/batchNorm6/beta:0').eval()),
'conv7/weights'             :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye1/conv7/weights:0').eval()),
'conv7/biases'              :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye1/conv7/biases:0').eval()),
'batchNorm7/gamma'          :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye1/batchNorm7/gamma:0').eval()),
'batchNorm7/beta'           :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye1/batchNorm7/beta:0').eval()),
'conv8/weights'             :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye1/conv8/weights:0').eval()),
'conv8/biases'              :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye1/conv8/biases:0').eval()),
'batchNorm8/gamma'          :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye1/batchNorm8/gamma:0').eval()),
'batchNorm8/beta'           :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye1/batchNorm8/beta:0').eval()),
'conv9/weights'             :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye1/conv9/weights:0').eval()),
'conv9/biases'              :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye1/conv9/biases:0').eval()),
'batchNorm9/gamma'          :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye1/batchNorm9/gamma:0').eval()),
'batchNorm9/beta'           :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye1/batchNorm9/beta:0').eval()),
'conv10/weights'            :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye1/conv10/weights:0').eval()),
'conv10/biases'             :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye1/conv10/biases:0').eval()),
'batchNorm10/gamma'         :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye1/batchNorm10/gamma:0').eval()),
'batchNorm10/beta'          :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye1/batchNorm10/beta:0').eval()),
'dense1/weights'            :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye1/dense1/weights:0').eval()),
'dense1/biases'             :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye1/dense1/biases:0').eval()),
'batchNorm11/gamma'         :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye1/batchNorm11/gamma:0').eval()),
'batchNorm11/beta'          :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye1/batchNorm11/beta:0').eval()),
'dense2/weights'            :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye1/dense2/weights:0').eval()),
'dense2/biases'             :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye1/dense2/biases:0').eval()),
'batchNorm12/gamma'         :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye1/batchNorm12/gamma:0').eval()),
'batchNorm12/beta'          :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye1/batchNorm12/beta:0').eval()),
'classify/weights'          :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye1/classify/weights:0').eval()),
'classify/biases'           :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye1/classify/biases:0').eval())
        }


    def __call__(self):
        return self.Dict





class eye2InitDictStage3:
    def __init__(self,
                 activeGraph,
                 weightsInit=tf.truncated_normal_initializer(stddev=0.04),
                 biasInit= tf.zeros_initializer(),gammaInit=None,betaInit=None):

        self.Dict = {
'conv1/weights'             :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye2/conv1/weights:0').eval()),
'conv1/biases'              :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye2/conv1/biases:0').eval()),
'batchNorm1/gamma'          :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye2/batchNorm1/gamma:0').eval()),
'batchNorm1/beta'           :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye2/batchNorm1/beta:0').eval()),
'conv2/weights'             :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye2/conv2/weights:0').eval()),
'conv2/biases'              :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye2/conv2/biases:0').eval()),
'batchNorm2/gamma'          :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye2/batchNorm2/gamma:0').eval()),
'batchNorm2/beta'           :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye2/batchNorm2/beta:0').eval()),
'conv3/weights'             :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye2/conv3/weights:0').eval()),
'conv3/biases'              :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye2/conv3/biases:0').eval()),
'batchNorm3/gamma'          :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye2/batchNorm3/gamma:0').eval()),
'batchNorm3/beta'           :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye2/batchNorm3/beta:0').eval()),
'conv4/weights'             :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye2/conv4/weights:0').eval()),
'conv4/biases'              :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye2/conv4/biases:0').eval()),
'batchNorm4/gamma'          :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye2/batchNorm4/gamma:0').eval()),
'batchNorm4/beta'           :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye2/batchNorm4/beta:0').eval()),
'conv5/weights'             :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye2/conv5/weights:0').eval()),
'conv5/biases'              :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye2/conv5/biases:0').eval()),
'batchNorm5/gamma'          :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye2/batchNorm5/gamma:0').eval()),
'batchNorm5/beta'           :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye2/batchNorm5/beta:0').eval()),
'conv6/weights'             :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye2/conv6/weights:0').eval()),
'conv6/biases'              :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye2/conv6/biases:0').eval()),
'batchNorm6/gamma'          :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye2/batchNorm6/gamma:0').eval()),
'batchNorm6/beta'           :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye2/batchNorm6/beta:0').eval()),
'conv7/weights'             :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye2/conv7/weights:0').eval()),
'conv7/biases'              :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye2/conv7/biases:0').eval()),
'batchNorm7/gamma'          :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye2/batchNorm7/gamma:0').eval()),
'batchNorm7/beta'           :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye2/batchNorm7/beta:0').eval()),
'conv8/weights'             :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye2/conv8/weights:0').eval()),
'conv8/biases'              :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye2/conv8/biases:0').eval()),
'batchNorm8/gamma'          :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye2/batchNorm8/gamma:0').eval()),
'batchNorm8/beta'           :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye2/batchNorm8/beta:0').eval()),
'conv9/weights'             :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye2/conv9/weights:0').eval()),
'conv9/biases'              :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye2/conv9/biases:0').eval()),
'batchNorm9/gamma'          :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye2/batchNorm9/gamma:0').eval()),
'batchNorm9/beta'           :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye2/batchNorm9/beta:0').eval()),
'conv10/weights'            :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye2/conv10/weights:0').eval()),
'conv10/biases'             :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye2/conv10/biases:0').eval()),
'batchNorm10/gamma'         :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye2/batchNorm10/gamma:0').eval()),
'batchNorm10/beta'          :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye2/batchNorm10/beta:0').eval()),
'dense1/weights'            :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye2/dense1/weights:0').eval()),
'dense1/biases'             :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye2/dense1/biases:0').eval()),
'batchNorm11/gamma'         :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye2/batchNorm11/gamma:0').eval()),
'batchNorm11/beta'          :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye2/batchNorm11/beta:0').eval()),
'dense2/weights'            :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye2/dense2/weights:0').eval()),
'dense2/biases'             :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye2/dense2/biases:0').eval()),
'batchNorm12/gamma'         :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye2/batchNorm12/gamma:0').eval()),
'batchNorm12/beta'          :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye2/batchNorm12/beta:0').eval()),
'classify/weights'          :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye2/classify/weights:0').eval()),
'classify/biases'           :   tf.constant_initializer(activeGraph.get_tensor_by_name('stage2/eye2/classify/biases:0').eval())
        }


    def __call__(self):
        return self.Dict







class BrainInitDictStage1:
    def __init__(self,
                 weightsInit=tf.truncated_normal_initializer(stddev=0.04),
                 biasInit= tf.zeros_initializer(),gammaInit=None,betaInit=None):
        self.Dict = {

'dense1/weights'     :   weightsInit,
'dense1/biases'      :   biasInit,
'batchNorm1/gamma'       :   gammaInit,
'batchNorm1/beta'        :   betaInit,

'dense2/weights'     :   weightsInit,
'dense2/biases'      :   biasInit,
'batchNorm2/gamma'       :   gammaInit,
'batchNorm2/beta'        :   betaInit,

'dense3/weights'     :   weightsInit,
'dense3/biases'      :   biasInit,
'batchNorm3/gamma'       :   gammaInit,
'batchNorm3/beta'        :   betaInit,

'classify/weights'        :   weightsInit,
'classify/biases'     :   biasInit
        }
    def __call__(self):
        return self.Dict

