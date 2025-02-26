import sys
from scipy.spatial.distance import pdist

def print_progress_bar(index, total, label):
    """
    prints progress bar for loops.

    :param index : current index of loop
    :param total : total number of indicies to loop over
    :param label : print statement next to progress bar

    """
    n_bar = 50  # Progress bar width
    progress = index / total
    sys.stdout.write('\r')
    sys.stdout.write(f"[{'=' * int(n_bar * progress):{n_bar}s}] {int(100 * progress)}%  {label}")
    sys.stdout.flush()


def get_l_prior(points):
    """
    Gives informative mean and std deviation for length scale priors
    for inverse gamma
    https://www.pymc.io/projects/examples/en/latest/gaussian_processes/GP-Heteroskedastic.html?fbclid=IwZXh0bgNhZW0CMTEAAR2qSLtP18yNHOB3M-_sUYPcpC6lGLFpm-fALLGbBXtnJTndooSsp2R7YuY_aem_utMnRz88GFKVRwu98TkY1w
    """
    
    distances = pdist(points)
    distinct = distances != 0
    lower = distances[distinct].min() if sum(distinct) > 0 else 0.1
    upper = distances[distinct].max() if sum(distinct) > 0 else 1
    std = max(0.1, (upper - lower) / 6)
    mu = lower + 3 * std
    return mu, std