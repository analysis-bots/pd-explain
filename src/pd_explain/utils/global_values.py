USE_SAMPLING = True


def toggle_sampling(new_value: bool = None):
    """
    Toggle the use of sampling on or off on the global level.
    If new_value is provided, set the use of sampling to that value.
    If new_value is not provided, toggle the use of sampling.
    :param new_value: The new value to set the use of sampling to.
    """
    global USE_SAMPLING
    if new_value is not None:
        USE_SAMPLING = new_value
    else:
        USE_SAMPLING = not USE_SAMPLING


def get_use_sampling_value():
    """
    Get the current value of the use of sampling.
    :return: The current value of the use of sampling.
    """
    return USE_SAMPLING