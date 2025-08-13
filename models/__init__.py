import importlib


def get_model(**kwargs):
    """
    Get model class from model module.
    Args:
        model_name (str): Name of the model file.
        class_name (str): Name of the model class.
        **kwargs: Additional keyword arguments to pass to the model class.
    Returns:
    """
    model_name = kwargs.get("model_name", None)
    class_name = kwargs.get("class_name", None)
    if model_name is None:
        print("Error: 'model_name' is required.")
        return
    
    module = importlib.import_module(f"models.{model_name}")
    
    model_class = getattr(module, class_name)  
    print(f"Model '{model_name}' found, using class '{class_name}'.")
    return model_class(**kwargs)    