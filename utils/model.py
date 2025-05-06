"""
Model related utility functions.
"""
import torch
from fvcore.nn import FlopCountAnalysis, parameter_count


def count_parameters(model, label="Model", print_result=True):
    """
    Prints the total and trainable number of parameters in a given PyTorch model.

    Useful  for analyzing model size, memory requirements, debugging, and optimization.

    Args:
        model (nn.Module): The PyTorch model whose parameters are to be counted.
        label (str): A label or name for the model (default is "Model"). Used in the print statement.
        print_result (bool): Whether to print the parameter count summary (default is True).

    Returns:
        tuple: (num_params, num_trainable_params), both as integers.
    """

    num_params = 0
    num_trainable_params = 0
    for param in model.parameters():
        num_params += (
            param.numel()
        )  # numel() returns the total number of elements in the tensor
        if param.requires_grad:
            num_trainable_params += param.numel()
    if print_result:
        print(
            f"Parameters number of {label}: {num_params / 1e6:.4f}M ({num_trainable_params / 1e6:.4f}M trainable)"
        )
    return num_params, num_trainable_params


def print_model_architecture(model):
    """
    Prints the architecture of a given PyTorch model using the standard `print` function.

    Args:
        model (nn.Module): The PyTorch model whose architecture is to be printed.

    Returns:
        None
    """
    print("Model Architecture:\n")
    print(model)


def get_model_flops(model, input_shape=(3, 224, 224), print_result=True):
    """
    Estimates the number of FLOPs (Floating Point Operations) for a given PyTorch model.

    Args:
        model (nn.Module): The PyTorch model whose FLOPs are to be estimated.
        input_shape (tuple): Shape of the input tensor (default is (3, 224, 224)).
        print_result (bool): Whether to print the FLOPs and parameter count (default is True).

    Returns:
        flops (int): Estimated number of FLOPs for the model.
    """
    dummy_input = torch.randn(1, *input_shape)
    flops = FlopCountAnalysis(model, dummy_input)
    params = parameter_count(model)
    total_flops = flops.total()
    total_params = params[""]

    if print_result:
        print(
            f"FLOPs: {total_flops / 1e9:.4f} GFLOPs \n Parameters: {total_params / 1e6:.4f}M"
        )
    return total_flops, total_params


def summarize_layer_params(model):
    """
    Summarizes the number of parameters in each layer of a given PyTorch model.

    Args:
        model (nn.Module): The PyTorch model whose layer parameters are to be summarized.

    Returns:
        dict: A dictionary where keys are layer names and values are the number of parameters in each layer.
    """
    layer_params = {}
    print("Layer Parameters Summary: \n")
    for name, param in model.named_parameters():
        layer_params[name] = param.numel()
        print(f"    - {name}: {param.numel() / 1e6:.4f}M")
    return layer_params


def freeze_layers(model, layers_to_freeze=[], print_result=True):
    """
    Freezes the parameters of specified layers in a given PyTorch model.

    Args:
        model (nn.Module): The PyTorch model whose layers are to be frozen.
        layers_to_freeze (List[str]): List of layer names to freeze (default is empty list).
        print_result (bool): Whether to print the names of frozen layers (default is True).

    Returns:
        None
    """
    for name, param in model.named_parameters():
        if any(layer in name for layer in layers_to_freeze):
            param.requires_grad = False
            if print_result:
                print(f"Freezing layer: {name}")


def freeze_all(model):
    """
    Freezes all parameters in the model to make them non-trainable.

    Args:
        model (nn.Module): The PyTorch model whose parameters are to be frozen.

    Returns:
        None
    """
    for param in model.parameters():
        param.requires_grad = False
    print("All layers are frozen.")


def unfreeze_all(model):
    """
    Unfreezes all parameters in the model to make them trainable.

    Args:
        model (nn.Module): The PyTorch model whose parameters are to be unfrozen.

    Returns:
        None
    """
    for param in model.parameters():
        param.requires_grad = True
    print("All layers are unfrozen.")
