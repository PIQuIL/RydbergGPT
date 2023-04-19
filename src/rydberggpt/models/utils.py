def extract_model_info(module, prefix=""):
    result = {}
    for name, submodule in module.named_children():
        if prefix:
            full_name = f"{prefix}.{name}"
        else:
            full_name = name

        result[full_name] = {
            "name": full_name,
            "class": submodule.__class__.__name__,
            # "input_shape": None,
            # "output_shape": None,
            "num_parameters": sum(p.numel() for p in submodule.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in submodule.parameters() if p.requires_grad
            ),
        }
        result.update(extract_model_info(submodule, prefix=full_name))
    return result
