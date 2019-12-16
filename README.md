# pytorch_imperative_quantization_tool

Quantization tool is a plugin which is used to help to use pytorch post-trainning quantization. 
# Features
## 1. Fallback layer auto-tuning
For some model, some layers with quantized op will reduce the model accuracy obviously which lead to the quantized model can not be applied in the real application. Our tool can automiticall search these layers and fallback these layers to be FP32 op to meet the accuracy goal.   
# Install
```
git clone https://github.com/liangan1/pytorch_imperative_quantization_tool.git
cd pytorch_imperative_quantization_tool
python setup.py install
```
# Usage 
```
from pytorch_quantization_tool import *
```

# API Specification 
```
def quantization_auto_tuning(model, run_fn, run_args, run_calibration,
                             calibration_args, metric = "top-1", relative_error = 0.01,
                             absolute_error = 0.01, relative_err_master = True,
                             fallback_op_types=DEFAULT_QUANTIZED_OP,
                             performance_fine_tuning=True):
    r"""
    The auto-tuning tool API for user.

    Args:
        model:    the model should already be prepared by first three steps in [post-training static quantization](https://pytorch.org/docs/stable/quantization.html)
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
```

```
def prepare_fallback_model(model, fallback_info_directory = "quantized_model"):
    r"""
    The auto-tuning tool API for user to prepare fallback quantized model.
    user can use this model to load quantized parameter

    Args:
        model:    the model should already be prepared by first two steps in
        model_directory: directory where store the fallback layer infomation qconfig.json
    """
```
# Design flow chart 
![image](https://github.com/liangan1/pytorch_imperative_quantization_tool/blob/master/images/Drawing39.jpg)

