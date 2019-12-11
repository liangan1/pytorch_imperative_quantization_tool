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

class DequantQuantWrapper(torch.nn.Module):
    r"""A wrapper class that wraps the input module, adds DeQuantStub and
    QuantStub and surround the call to module with call to dequant and quant
    modules.

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

class SaveTensorObserver(torch.quantization.observer._ObserverBase):
    r"""
    The module is mainly for debug and records the tensor values during runtime.

    Args:
        layer_name: model layer name format is : xxx.xxx.xxx
        saved: flag to save, only save the first one batch tensor
    """
    def __init__(self, layer_name = ""):
        super(SaveTensorObserver, self).__init__()
        self.saved = False
        self.layer_name = layer_name
    def forward(self, x):
        if self.saved:
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
        self.saved = True

    @torch.jit.export
    def calculate_qparams(self):
        raise Exception("calculate_qparams should not be called for SaveTensorObserver")

def add_save_observer_(module, prefix = "", fallback_op_types=DEFAULT_QUANTIZED_OP):
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
            add_save_observer_(child, prefix + name + ".")

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

def fallback_layer(model, layer_name="", exculde_layers={}):
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
           model._modules[name] = DequantQuantWrapper(sub_model)
        else:
           fallback_layer(sub_model, sub_model_layer_name, exculde_layers)

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

def save_quantized_model(model, fallback_layers, save_directory="quantized_model", save_config = False):
    r"""
    save quantized model info:
    1) fallback_layer info
    2) configuration if need, sunch bert model
    3) quantized_model state_dict

    Args:
        model: quantized model
        fallback_layers: layers force to be fp32 op
        save_directory:  directory to save model info
        save_config: if need to save configuration information
    """

    if not os.path.exists(save_directory):
       os.mkdir(save_directory)
    assert os.path.isdir(save_directory), "Saving path should be a directory where the model and configuration can be saved"
    # Save qconfig info
    qconfig_file = os.path.join(save_directory, "qconfig.json")
    with open(qconfig_file, "w") as qconfig_output:
         json.dump(fallback_layers, qconfig_output)
    # Save configuration file (reference to pytorch_transformers repo)
    if save_config:
       config_file = os.path.join(save_directory, "config.json")
       with open(config_file, "w", encoding='utf-8') as writer:
            output = copy.deepcopy(model.config.__dict__)
            json_str = json.dumps(output, indent=2, sort_keys=True) + "\n"
            writer.write(json_str)
    # Only save the model it-self if we are using distributed training
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(save_directory, "pytorch_model.bin")
    torch.save(model_to_save.state_dict(), output_model_file)

def quantization_auto_tuning(model, run_fn, run_args, run_calibration,
                             calibration_args, metric = "top-1", relative_error = 0.01,
                             absolute_error = 0.01, relative_err_master = True,
                             fallback_op_types=DEFAULT_QUANTIZED_OP,
                             performance_fine_tuning=True):
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

    """
    #run fp32 evaluation to collect accuracy and fp32 tensor
    model_tmp = copy.deepcopy(model)
    propagate_qconfig_(model_tmp)
    add_save_observer_(model_tmp)
    result = run_fn(model_tmp, run_args)
    fp32_accuracy = result[metric]
    #run calibration
    model_tmp = copy.deepcopy(model)
    prepare(model_tmp, inplace = True)
    run_calibration(model_tmp, calibration_args)

    #run int8 to collect accuracy and int8 tensor
    convert(model_tmp, inplace=True)
    add_save_observer_(model_tmp)
    result = run_fn(model_tmp, run_args)
    int8_accuracy = result[metric]
    save_quantized_model(model_tmp, {},
                            save_directory="quantized_model", save_config = True)
    need_to_fallback = False
    if relative_err_master:
       need_to_fallback = True if int8_accuracy < fp32_accuracy * (1 - relative_error) else False
    else:
       need_to_fallback = True if int8_accuracy < fp32_accuracy * (1 - absolute_error) else False

    #begin to fallback auto-tuning
    if need_to_fallback:
       #comput distance between fp32 tensor and int8 dequantize tensor
       layer_gap_dict = {}
       compute_fp32_and_int8_dequantize_gap(model, "", layer_gap_dict)
       #sort layer according to above distance to construct auto-tuning search order
       sorted_gap = sorted(layer_gap_dict.items(), key=lambda item:item[1], reverse=True)
       for item in sorted_gap:
           print(item)

       cur_int8_accuracy = int8_accuracy
       pre_int8_accuracy = int8_accuracy #the currenty best accuacy
       len_gap_dict = len(layer_gap_dict)#the maximum search times
       fallback_layers = {} #bucket to save fallback layers
       accuracy_improvment_dict = {}
       count = 0
       #fallback auto-tuning
       while need_to_fallback and  count < len_gap_dict:
             #fallback layers in the bucket
             model_tmp = copy.deepcopy(model)
             propagate_qconfig_(model_tmp)
             fallback_layers.update({sorted_gap[count % len_gap_dict][0]:False})
             fallback_layer(model_tmp, "", fallback_layers)

             #calibration and validate the accuracy of
             #partitial fallback quantized model_tmp
             add_observer_(model_tmp)
             run_calibration(model_tmp, calibration_args)
             convert(model_tmp, inplace = True)
             result = run_fn(model_tmp, run_args)
             cur_int8_accuracy=result[metric]
             if cur_int8_accuracy > pre_int8_accuracy:
                accuracy_improvment_dict.update(
                       {sorted_gap[count % len_gap_dict][0]:
                        cur_int8_accuracy - pre_int8_accuracy })
                print("accuracy_improvment_dict", accuracy_improvment_dict)
                pre_int8_accuracy = cur_int8_accuracy
             else:
                del fallback_layers[sorted_gap[count % len_gap_dict][0]]
             count += 1
             if relative_err_master:
                need_to_fallback = True if pre_int8_accuracy < fp32_accuracy * (1 - relative_error) else False
             else:
                need_to_fallback = True if pre_int8_accuracy < fp32_accuracy * (1 - absolute_error) else False
       print(performance_fine_tuning)
       performance_fine_tuning=True
       if performance_fine_tuning:
          #furtherly search the  subset of fallback_layers to improve performance
          fined_fallback_layers = {}
          #sort layer by accuracy value difference
          fallback_layers = sorted(fallback_layers.items(),
                            key=lambda item:item[1], reverse=True)
          print("Candidate fallback_layers:")
          for item in fallback_layers:
              print(item)
          for layer in fallback_layers:
              print(type(layer))
              model_tmp = copy.deepcopy(model)
              propagate_qconfig_(model_tmp)
              fined_fallback_layers.update({layer[0]:layer[1]})
              fallback_layer(model_tmp, "", fined_fallback_layers)

              #calibration and validate the accuracy of
              #partitial fallback quantized model_tmp
              add_observer_(model_tmp)
              run_calibration(model_tmp, calibration_args)
              convert(model_tmp, inplace = True)
              result = run_fn(model_tmp, run_args)
              cur_int8_accuracy=result[metric]
              if relative_err_master and cur_int8_accuracy >= fp32_accuracy * (1 - relative_error):
                 break
              elif not relative_err_master and cur_int8_accuracy >= fp32_accuracy * (1 - absolute_error):
                 break

       if performance_fine_tuning:
          fallback_layers = fined_fallback_layers

       propagate_qconfig_(model)
       fallback_layer(model, "", fallback_layers)

       #calibration and validate the accuracy of
       #partitial fallback quantized model
       add_observer_(model)
       run_calibration(model, calibration_args)
       convert(model, inplace = True)
       result = run_fn(model, run_args)
       print("The fallback layers as following:")
       for layer in fallback_layers.keys():
           print(layer)
       print("The Int8 accuacy:", result)
       save_quantized_model(model, fallback_layers=fallback_layers,
                            save_directory="quantized_model", save_config = True)
                                    
