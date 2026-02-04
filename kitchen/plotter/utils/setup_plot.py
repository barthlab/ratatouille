import matplotlib.pyplot as plt


def apply_plot_settings(ax: plt.Axes, settings_dict: dict):
    """
    Applies plot settings to a matplotlib Axes object from a dictionary.

    Args:
        ax (matplotlib.axes.Axes): The Axes object to modify.
        settings_dict (dict): A dictionary where keys are the names of 
                              setter methods (e.g., 'set_xlabel', 'set_title')
                              and values are the arguments for those methods.
    """
    for key, value in settings_dict.items():
        if hasattr(ax, key):
            if isinstance(value, (list, tuple)):
                getattr(ax, key)(*value)
            elif isinstance(value, dict):
                getattr(ax, key)(**value)
            else:
                getattr(ax, key)(value)
        elif key == "_self" and callable(value):
            value(ax)
        else:
            raise AttributeError(f"Invalid attribute or method for Axes: {key}. Cannot apply setting.")
