from matplotlib.colors import ListedColormap

def get_pastel():
    """
    Returns my_pastel colormap.
    """
    # Custom color map 
    custom_colors = [
        'midnightblue',
        'royalblue',
        'skyblue',
        'teal',
        'cadetblue',
        'darkseagreen',
        'darkred',
        'firebrick',
        'indianred',
        'darkorchid',
        'mediumorchid',
        'orchid',
        'darkolivegreen',
        'olivedrab',
        'yellowgreen',
        'darkgoldenrod',
        'goldenrod',
        'darkkhaki'
    ]
    return ListedColormap(custom_colors, name='my_pastel')