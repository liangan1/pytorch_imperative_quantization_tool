import torch 
from torch.quantization.quantize import _observer_forward_hook
from torch.quantization import \
    prepare, convert, propagate_qconfig_, add_observer_
from torch.quantization import \
    QuantWrapper, QuantStub, DeQuantStub, default_qconfig, default_per_channel_qconfig
import torch.nn.intrinsic.quantized as nniq
import torch.nn.quantized as nnq
import copy
import numpy as np
import json
import os 


DEFAULT_QUANTIZED_OP = {
    nnq.Linear,
    nnq.ReLU,
    nnq.ReLU6,
    nnq.Conv2d,
    # Wrapper Modules:
    nnq.QFunctional,
    # Intrinsic modules:
    nniq.ConvReLU2d,
    nniq.LinearReLU,
    nniq.ConvReLU2d,
    nniq.LinearReLU,
    nnq.Conv2d,
    nniq.ConvReLU2d,
}

STR2QCONFIG = {
    "default_qconfig" : default_qconfig,
    "default_per_channel_qconfig" : default_per_channel_qconfig
}

QCONFIG2STR = {
    default_qconfig : "default_qconfig",
    default_per_channel_qconfig : "default_per_channel_qconfig"
}

class DequantQuantWrapper(torch.nn.Module):
    r"""A wrapper class that wraps the input module, adds DeQuantStub and
    surround the call to module with call to dequant.
    this is used by fallback layer when the data type of quantized op 
    is  input:int8/output:int8. 

    This is used by the fallback utility functions to add the dequant and
    quant modules, before `convert` function `QuantStub` will just be observer,
    it observes the input tensor, after `convert`, `QuantStub`
    will be swapped to `nnq.Quantize` which does actual quantization. Similarly
    for `DeQuantStub`.
    """
    def __init__(self, module):
        super(DequantQuantWrapper, self).__init__()
        self.add_module('quant', QuantStub(module.qconfig))
        self.add_module('dequant', DeQuantStub())
        self.add_module('module', module)
        module.qconfig = None
        self.train(module.training)

    def forward(self, X):
        X = self.dequant(X)
        X = self.module(X)
        return self.quant(X)

class DequantWrapper(torch.nn.Module):
    r"""A wrapper class that wraps the input module, adds DeQuantStub and
    surround the call to module with call to dequant modules. this is used by fallback
    layer when the data type of quantized op is  input:int8/output:fp32

    This is used by the fallback utility functions to add the dequant and
    quant modules, before `convert` function `QuantStub` will just be observer,
    it observes the input tensor, after `convert`, `DeQuantStub`
    will be swapped to `nnq.DeQuantize` which does actual dequantization. 
    """
    def __init__(self, module):
        super(DequantQuantWrapper, self).__init__()
        self.add_module('dequant', DeQuantStub())
        self.add_module('module', module)
        module.qconfig = None
        self.train(module.training)

    def forward(self, X):
        X = self.dequant(X)
        return self.module(X)


class SaveTensorObserver(torch.quantization.observer._ObserverBase):
    r"""
    The module is mainly for debug and records the tensor values during runtime.

    Args:
        layer_name: model layer name format is : xxx.xxx.xxx
        saved: flag to save, only save the first one batch tensor
    """
    def __init__(self, layer_name = "", saved = True):
        super(SaveTensorObserver, self).__init__()
        self.saved = saved
        self.layer_name = layer_name
    def forward(self, x):
        if not self.saved:
           return
        if x.is_quantized:
           INT8_tesor_path = "INT8_tesor/"
           if not os.path.exists(INT8_tesor_path):
              os.mkdir(INT8_tesor_path)
           np.save(INT8_tesor_path + self.layer_name, x.dequantize())
           print("saved quantized tensor ", self.layer_name)
        else:
           FP32_tesor_path = "FP32_tesor/"
           if not os.path.exists(FP32_tesor_path):
              os.mkdir(FP32_tesor_path)
           np.save(FP32_tesor_path + self.layer_name, x)
           print("saved FP32 tensor ", self.layer_name)
        self.saved = False

    @torch.jit.export
    def calculate_qparams(self):
        raise Exception("calculate_qparams should not be called for SaveTensorObserver")

def add_save_observer_(module, prefix = "", saved=True, fallback_op_types=DEFAULT_QUANTIZED_OP):
    r"""Add save tensor observer for some conditional leaf child of the module.

    Args:
        module: input module

    Return:
        None, module is modified inplace with added observer modules and forward_hooks
    """
    for name, child in module.named_children():
        if type(child) == nnq.FloatFunctional:
            if hasattr(child, 'qconfig') and child.qconfig is not None:
                child.activation_post_process = SaveTensorObserver(prefix+name+".")
        else:
            add_save_observer_(child, prefix + name + ".", saved=saved, fallback_op_types=fallback_op_types)

    # Insert observers only for leaf nodes, note that this observer is for
    # the output of the module, for input QuantStub will observe them
    if len(module._modules) == 0 and (type(module) in DEFAULT_QUANTIZED_OP \
       or hasattr(module, 'qconfig') and module.qconfig is not None and \
       not isinstance(module, torch.nn.Sequential) and not isinstance(module,  DeQuantStub) and \
        not isinstance(module, QuantStub)):
        print(type(module))
        print( isinstance(module,  DeQuantStub))
        # observer and hook will be gone after we swap the module
        module.add_module('activation_post_process', SaveTensorObserver(prefix))
        module.register_forward_hook(_observer_forward_hook)

def mse_metric_gap(fp32_tensor, int8_dequantize_tensor):
    r"""
        caculate the euclidean distance between
        fp32 tensor and int8 dequantize tensor

    Args:
        fp32_tensr:
        int8_dequantize_tensor:
    """
    fp32_max  = np.max(fp32_tensor)
    fp32_min  = np.min(fp32_tensor)
    int8_dequantize_max = np.max(int8_dequantize_tensor)
    int8_dequantize_min = np.min(int8_dequantize_tensor)
    fp32_tensor = (fp32_tensor - fp32_min) / (fp32_max - fp32_min)
    int8_dequantize_tensor = (int8_dequantize_tensor - int8_dequantize_min) / (int8_dequantize_max - int8_dequantize_min)
    diff_tensor = fp32_tensor - int8_dequantize_tensor
    euclidean_dist = np.sum(diff_tensor ** 2)
    return euclidean_dist/fp32_tensor.size

def fallback_layer(model, layer_name="", exculde_layers={}, max_split_quantized=False):
    r"""
    force layers in exculde_layers bucket to be fp32 op, the model
    should be add qconfig and QuantWrapper already

    Args:
        model: a fp32 model with qconfig information
        layer_name: model layer name format is : xxx.xxx.xxx
        : flag to save, only save the first one batch tensor
    """

    for name, sub_model in list(model.named_children()):
        sub_model_layer_name = layer_name + name + "."
        if sub_model_layer_name in exculde_layers:
           print("fallback_layer:", sub_model_layer_name)
           if max_split_quantized:
               sub_model.qconfig = None
               for  name_tmp, sub_model_tmp in list(model.named_children()):
                    if (isinstance(sub_model_tmp, QuantStub) or isinstance(sub_model_tmp, DeQuantStub)):
                       model._modules[name_tmp]=torch.nn.Identity()            
           else:
               model._modules[name] = DequantQuantWrapper(sub_model)
        else:
           fallback_layer(sub_model, sub_model_layer_name, exculde_layers, max_split_quantized)

def compute_fp32_and_int8_dequantize_gap(model, layer_name="", layer_gap_dict={}):
    r"""
    load FP32_tesor and int8_dequantize_tensor, then compute the distance between
    the distance one by one.

    Args:
        layer_name: model layer name format is : xxx.xxx.xxx
        layer_gap_dict: a dict to save {layer_name: distance}
    """

    fp32_file = "FP32_tesor/" + layer_name + ".npy"
    int8_dequantize_file = "INT8_tesor/" + layer_name+".npy"
    if os.path.exists(fp32_file) and os.path.exists(int8_dequantize_file):
       print(layer_name)
       fp32_tensor = np.load(fp32_file)
       int8_dequantize_tensor = np.load(int8_dequantize_file)
       gap = mse_metric_gap(fp32_tensor, int8_dequantize_tensor)
       layer_gap_dict.update({layer_name:gap})
    for name, sub_model in model.named_children():
        compute_fp32_and_int8_dequantize_gap(sub_model, layer_name + name + ".", layer_gap_dict)

def save_quantized_model(model, qconfig=None, fallback_layers={}, save_directory="quantized_model"):
    r"""
    save quantized model info:
    1) fallback_layer info
    2) configuration if need, sunch bert model
    3) quantized_model state_dict

    Args:
        model: quantized model
        fallback_layers: layers force to be fp32 op
        qconfig: QConfig used for quantized model 
        save_directory:  directory to save model info
        save_config: if need to save configuration information
    """

    assert qconfig is not None, "qconfig can not be None"
    assert qconfig in QCONFIG2STR, "qconfig must be in QCONFIG2STR dict"

    qconfig_dict={"qconfig":QCONFIG2STR[qconfig]}
    qconfig_dict.update({"fallback_layers": fallback_layers})

    if not os.path.exists(save_directory):
       os.mkdir(save_directory)
    assert os.path.isdir(save_directory), "Saving path should be a directory where the model and configuration can be saved"
    # Save qconfig info
    qconfig_file = os.path.join(save_directory, "qconfig.json")
    with open(qconfig_file, "w") as qconfig_output:
         json.dump(qconfig_dict, qconfig_output)
    # Save configuration file (reference to pytorch_transformers repo)
    if hasattr(model, "config"):
       config_file = os.path.join(save_directory, "config.json")
       with open(config_file, "w", encoding='utf-8') as writer:
            output = copy.deepcopy(model.config.__dict__)
            json_str = json.dumps(output, indent=2, sort_keys=True) + "\n"
            writer.write(json_str)
   
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(save_directory, "pytorch_model.bin")
    torch.save(model_to_save.state_dict(), output_model_file)

def prepare_fallback_model(model, quantized_model_directory = "quantized_model", max_split_quantized=False):
    r"""
    The auto-tuning tool API for user to prepare fallback quantized model. 

    Args:
        model:    the model should already be prepared by first two steps in    
        model_directory: directory where store the fallback layer infomation qconfig.json 
        max_plit_quantized: every quantized op is surrounded by QuantStub and DeQuantStub,and only one
                             quantized op in every level module, such as BERT in pytorch_transformers.

    """
    qconfig_file = os.path.join(quantized_model_directory, "qconfig.json")
    with open(qconfig_file) as f:
         qconfig_dict = json.load(f)
         model.qconfig = STR2QCONFIG[qconfig_dict["qconfig"]]
         fallback_layers = qconfig_dict["fallback_layers"]
    propagate_qconfig_(model)
    if len(fallback_layers) > 0:
       fallback_layer(model, "", fallback_layers, max_split_quantized)
    add_observer_(model)
    convert(model, inplace = True)
    quantized_model_file = os.path.join(quantized_model_directory, "pytorch_model.bin")  
    state_dict = torch.load(quantized_model_file)  
    model.load_state_dict(state_dict)
    return model 

def get_original_quantized_layer(model, layer_name="", layers=[], fallback_op_types=DEFAULT_QUANTIZED_OP):
    r"""
    when the tuing_strategy is bottom-up, we firstly use this API to get all original quantized layers.

    Args:
    model:The quantized model 
    layer_name: the name of layers speperated by "."
    layers: original quantized layers
    fallback_op_types: opreator types which you want to fallback  
    
    """
    for name, child in model.named_children():
        sub_model_layer_name = layer_name + name + "."
        if len(child._modules) == 0 and type(child) in DEFAULT_QUANTIZED_OP:
           layers.append(sub_model_layer_name)
        else:
           get_original_quantized_layer(child, sub_model_layer_name, layers, fallback_op_types)

def run(model, run_fn, run_args, run_calibration=None, calibration_args=None, qconfig=None, metric="acc",
        fallback_layers=None, tuing_strategy="bottom-up", 
        save_fp32_tensor=False, save_int8_tensor=False, max_split_quantized=False):

    model_tmp = copy.deepcopy(model)

    #run fp32 evaluation to collect accuracy and fp32 tensor
    if save_fp32_tensor:
       propagate_qconfig_(model_tmp)
       add_save_observer_(model_tmp)

    #run calibration
    if qconfig is not None:
       model.qconfig = qconfig
       propagate_qconfig_(model_tmp)
       if fallback_layers is not None:
          fallback_layer(model_tmp, "", fallback_layers, max_split_quantized)

       add_observer_(model_tmp)
       run_calibration(model_tmp, calibration_args)
       convert(model_tmp, inplace = True)
       if save_int8_tensor is True and qconfig == default_per_channel_qconfig:
          add_save_observer_(model_tmp)

    #run_model
    result = run_fn(model_tmp, run_args)
    accuracy=result[metric]
    return model_tmp, accuracy

def accuracy_is_meet_goal(fp32_accuracy=0.0, int8_accuracy=0.0, relative_err_master=True, 
                          relative_error=0.01, absolute_error=0.01):

    meet_goal = False
    if relative_err_master:
       meet_goal = False if int8_accuracy < fp32_accuracy * (1 - relative_error) else True
    else:
       meet_goal = False if int8_accuracy < fp32_accuracy * (1 - absolute_error) else True

    return meet_goal

def quantization_auto_tuning(model, run_fn, run_args, run_calibration,
                             calibration_args, metric = "top-1", relative_error = 0.01,
                             absolute_error = 0.01, relative_err_master = True,
                             quantized_model_directory = "quantized_model",
                             fallback_op_types=DEFAULT_QUANTIZED_OP,
                             performance_fine_tuning=True,
                             max_split_quantized=False,
                             tuing_strategy="bottom-up",
                             save_config = True):
    r"""
    The auto-tuning tool API for user.

    Args:
        model:    the model should already be prepared by first two steps in
        run_fn:   evaluation function, the return should be {accuracy_metric:value}
                  for example, {"acc": 0.62}
        run_args: this is the args of evaluation function, recommond using
                  the type of parser.parse_args()
        run_calibration: calibration function
        calibration_args: the args for calibration function
        tuing_strategy: bttom-up or euclidean
        metric:   the accuracy metric, such as: acc, f1, mcc, top-1, map and so.
        relative_error: the maximum torlerance ratio of relative error btween fp32 model
                        and quantized model, the default value is 0.01 (1%)
        absolute_error: the maximum torlerance ratio of absolute error btween fp32 model
                        and quantized model, the default value is 0.01 (1%)
        relative_err_master: whether relative_error or absolute_error is import for you
        fallback_op_types: which type quantized op should be auto-tuing fallback, there
                           are generally several diffrent quantized op in the quantized
                           model, sometimes, you just want to fallback some types not all types.
                           for example: conv/linear are in a CV model, you just want to fallback
                           linear, then fallback_op_types={nnq.Linear}
        max_split_quantized: every quantized op is surrounded by QuantStub and DeQuantStub,and only one 
                             quantized op in every level module, such as BERT in pytorch_transformers.

    """
    original_quantized_layers = []
    fp32_accuracy = 0.0
    int8_accuracy = 0.0
    save_fp32_tensor = False
    save_int8_tensor = False
    qconfig = None
    quantized_model = None
    need_to_fallback = True    
 
    if tuing_strategy == "euclidean":
       save_fp32_tensor = True
       save_int8_tensor = True
       
    _, fp32_accuracy = run(model, run_fn=run_fn, run_args=run_args, metric=metric, save_fp32_tensor=save_fp32_tensor)

    #begin to auto-tuning qconfig 
    for qconfig_item in QCONFIG2STR:
        print("####Tuning qconfig:", QCONFIG2STR[qconfig_item])
        quantized_model, cur_int8_accuracy = run(model, run_fn=run_fn, run_args=run_args, run_calibration=run_calibration, 
                                             calibration_args=calibration_args, qconfig=qconfig_item, metric=metric, 
                                             save_int8_tensor=save_int8_tensor)
        if qconfig is None or cur_int8_accuracy > int8_accuracy:
           int8_accuracy = cur_int8_accuracy
           qconfig = qconfig_item 
        accuracy_meet_goal = accuracy_is_meet_goal(fp32_accuracy,int8_accuracy,relative_err_master,relative_error,
                                                  absolute_error) 
        if accuracy_meet_goal:
           need_to_fallback = False
           save_quantized_model(quantized_model,  qconfig=qconfig, save_directory=quantized_model_directory)
           break 
    get_original_quantized_layer(quantized_model, layer_name ="", layers = original_quantized_layers, fallback_op_types=fallback_op_types)
    original_quantized_layers.reverse()
    
    print("####The best qconfig:", qconfig, "\nBase Acuuracy:", int8_accuracy)
    #begin to fallback auto-tuning
    if need_to_fallback:
       if tuing_strategy == "euclidean":
           #comput distance between fp32 tensor and int8 dequantize tensor
           layer_gap_dict = {}
           compute_fp32_and_int8_dequantize_gap(model, "", layer_gap_dict)
           #sort layer according to above distance to construct auto-tuning search order
           sorted_gap = sorted(layer_gap_dict.items(), key=lambda item:item[1], reverse=True)
           original_quantized_layers = [item[0] for item in sorted_gap]
      
       print("####fallback search order:")
       for item in original_quantized_layers:
           print(item)

       pre_int8_accuracy = int8_accuracy #the currenty best accuacy
       cur_int8_accuracy = pre_int8_accuracy
       len_gap_dict = len(original_quantized_layers)#the maximum search times
       fallback_layers = {} #bucket to save fallback layers
       accuracy_improvment_dict = {}
       count = 0
       #fallback auto-tuning
       while need_to_fallback and  count < len_gap_dict:
             #fallback layers in the bucket
             if tuing_strategy == "euclidean":
                fallback_layers.update({original_quantized_layers[count % len_gap_dict]:False})                
             elif tuing_strategy == "bottom-up":
                fallback_layers = {original_quantized_layers[count % len_gap_dict]:False}
                print(fallback_layers)
             _, cur_int8_accuracy = run(model, run_fn=run_fn, run_args=run_args, run_calibration=run_calibration,
                                        calibration_args=calibration_args, qconfig=qconfig, metric=metric, 
                                        fallback_layers=fallback_layers,max_split_quantized=max_split_quantized)
             if tuing_strategy == "euclidean":
                if cur_int8_accuracy > int8_accuracy:
                   accuracy_improvment_dict.update(
                       {original_quantized_layers[count % len_gap_dict]:
                        cur_int8_accuracy - int8_accuracy })
                   pre_int8_accuracy = cur_int8_accuracy
                else:
                    del fallback_layers[original_quantized_layers[count % len_gap_dict]]
             elif tuing_strategy == "bottom-up":
                if cur_int8_accuracy > int8_accuracy:
                   accuracy_improvment_dict.update(
                       {original_quantized_layers[count % len_gap_dict]:
                        cur_int8_accuracy - int8_accuracy })
                   pre_int8_accuracy = cur_int8_accuracy
             count += 1
             need_to_fallback = not accuracy_is_meet_goal(fp32_accuracy, cur_int8_accuracy, relative_err_master,
                                                          relative_error, absolute_error) 
       #furtherly search the  subset of fallback_layers to improve performance
       if performance_fine_tuning:
          fined_fallback_layers = {}
          #sort layer by accuracy value difference
          candidate_fallback_layers = sorted(accuracy_improvment_dict.items(),
                            key=lambda item:item[1], reverse=True)
          print("####Candidate fallback_layers:")
          for item in candidate_fallback_layers:
              print(item)
          pre_int8_accuracy = int8_accuracy
          for layer in candidate_fallback_layers:
              fined_fallback_layers.update({layer[0]:layer[1]})
              _, cur_int8_accuracy = run(model, run_fn=run_fn, run_args=run_args, run_calibration=run_calibration,
                                         calibration_args=calibration_args, qconfig=qconfig, metric=metric, 
                                         fallback_layers=fined_fallback_layers, max_split_quantized=max_split_quantized)
              if cur_int8_accuracy > pre_int8_accuracy:
                 pre_int8_accuracy = cur_int8_accuracy
              else:
                 del fined_fallback_layers[layer[0]]
              accuracy_meet_goal = accuracy_is_meet_goal(fp32_accuracy, cur_int8_accuracy, relative_err_master,
                                                          relative_error, absolute_error)
              if accuracy_meet_goal:
                 break

       if performance_fine_tuning:
          fallback_layers = fined_fallback_layers
       
       quantized_model, final_int8_accuracy = run(model, run_fn=run_fn, run_args=run_args, run_calibration=run_calibration, 
                                                  calibration_args=calibration_args, qconfig=qconfig, metric=metric,                                                                                                                        fallback_layers=fallback_layers, max_split_quantized=max_split_quantized)
       print("####The fallback layers as following:")
       for layer in fallback_layers.keys():
           print(layer)
       print("####The Int8 accuacy:", final_int8_accuracy)
       print(quantized_model)
       save_quantized_model(quantized_model, fallback_layers=fallback_layers, qconfig=qconfig,
                            save_directory=quantized_model_directory)
                                    
