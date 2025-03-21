import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path
from typing import Union
from astropy.io import fits
from astropy.visualization import ZScaleInterval, LinearStretch, ImageNormalize
from matplotlib.colors import LogNorm, PowerNorm

from pipnick.pipelines.reduction import init_ccddata
from pipnick.utils.fits_class import Fits_Simple
from pipnick.utils.dir_nav import unzip_directories


def print_fits_info(image_path: str):
    """
    Print HDU List information and display the FITS image data.

    Parameters
    ----------
    image_path : str
        Path to the FITS image (greyscale only).
    """
    with fits.open(image_path) as hdul:
        print("\nHDU Header")
        print(repr(hdul[0].header))
        
        plt.figure(figsize=(8, 6))
        interval = ZScaleInterval()
        vmin, vmax = interval.get_limits(hdul[0].data)
        plt.imshow(hdul[0].data, origin='lower', vmin=vmin, vmax=vmax)
        plt.gcf().set_dpi(300)
        plt.colorbar()
        plt.show()


def display_nickel(image: Union[str, Path, Fits_Simple]):
    """
    Display the data of a FITS image using zscale coloring, with masked
    pixels colored red.

    Parameters
    ----------
    image : Union[str, Path, Fits_Simple]
        The Fits_Simple object or path to the FITS image.
    """
    if not isinstance(image, Fits_Simple):
        image = Fits_Simple(image)

    data_masked = image.masked_array
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.set_title(image)

    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(data_masked)
    cmap = plt.get_cmap()
    cmap.set_bad('r', alpha=0.5)
    ax.imshow(data_masked, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(cm.ScalarMappable(cmap=cmap), ax=ax)
    plt.show()


def display_many_nickel(path_list):
    """
    Display the data of all images in a list of directories or files.

    Parameters
    ----------
    path_list : list of str
        A list of directory or file paths containing FITS images.
    """
    images = unzip_directories(path_list, output_format='Fits_Simple')
    for image in images:
        display_nickel(image)


def fits_plot(ccd, title='FITS Plot', 
              xlabel='Pixel X', ylabel='Pixel Y', cmap='gray', cbar_label='Counts (electrons)',
              vmin=1, vmax=99.75, figsize=(10, 10), mask_highlight=False):
    """
    Plots a FITS image using percentile scaling for contrast adjustment.
    Optionally, masked pixels (if available in ccd.mask) are highlighted in red.

    Parameters:
        ccd (object): An object containing the FITS image data (ccd.data) and an optional mask (ccd.mask).
        title (str): The title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        cmap (str): Colormap used in the plot.
        cbar_label (str): Label for the colorbar.
        vmin (float): Lower percentile for contrast scaling.
        vmax (float): Upper percentile for contrast scaling.
        figsize (tuple): Figure size in inches.
        mask_highlight (bool): If True, highlights masked pixels in red. Default is False.
    """
    data = ccd.data
    # Calculate display range using percentiles.
    lower = np.percentile(data, vmin)
    upper = np.percentile(data, vmax)
    
    # Create the figure.
    plt.figure(figsize=figsize)
    # Store the image mappable.
    im = plt.imshow(data, cmap=cmap, origin='lower', vmin=lower, vmax=upper)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    
    # Overlay red markers for masked pixels if mask_highlight is True.
    if mask_highlight and hasattr(ccd, 'mask') and ccd.mask is not None:
        masked_y, masked_x = np.where(ccd.mask)
        plt.scatter(masked_x, masked_y, s=10, facecolors='none', edgecolors='red', linewidths=0.5)
    
    # Create and label the colorbar explicitly using the image mappable.
    cbar = plt.colorbar(im, shrink=0.8)
    cbar.set_label(cbar_label)
    plt.show()


def get_array(data):
    """
    Returns the underlying NumPy array.
    
    If 'data' is a file path (str or Path), it is assumed to be a FITS file and is
    initialized using init_ccddata, then the underlying data array is returned.
    
    If 'data' is a CCDData object (or similar) with a .data attribute, returns that.
    Otherwise, assumes data is already a NumPy array.
    """
    if isinstance(data, (str, Path)):
        # If the input is a file path, initialize the CCDData using init_ccddata.
        data = init_ccddata(data)
    return data.data if hasattr(data, 'data') else data


def fits_plot_multi(data_list, titles=None,
                    xlabel='Pixel X', ylabel='Pixel Y',
                    cmap='gray', cbar_label='Counts (electrons)',
                    vmin=1, vmax=99.75, ncols=2, figsize=(15, 10),
                    share_axes=True, share_vminmax=False,
                    subplot_dims=None,
                    scale='linear', gamma=1.0):
    """
    Plots multiple FITS images in a grid of subplots using various scaling options.
    Each subplot is given its own colorbar, shrunk so it isn't too tall.
    
    Parameters:
        data_list (list): List of 2D arrays, CCDData objects, or FITS file paths.
        titles (list, optional): Titles for each subplot. Defaults to 'FITS Plot' if not provided.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        cmap (str): Colormap for the images.
        cbar_label (str): Label for the colorbar.
        vmin (float): Lower percentile (or lower limit for non-linear scales) for contrast scaling.
        vmax (float): Upper percentile (or upper limit for non-linear scales) for contrast scaling.
        ncols (int): Number of columns in the subplot grid (if subplot_dims not provided).
        figsize (tuple): Size of the entire figure.
        share_axes (bool): If True, subplots share the same x and y axes.
        share_vminmax (bool): If True, compute a global vmin/vmax (or equivalent) across all images.
        subplot_dims (list, optional): A two-element list [nrows, ncols] to explicitly set the grid dimensions.
        scale (str): Scaling option; choices: 'linear', 'log', 'power', 'sqrt', 'zscale'.
        gamma (float): Gamma value for 'power' scaling (ignored for other scales).
    """
    
    # Helper function: ensure a small positive value if needed for LogNorm
    def safe_lower(arr, lower):
        if lower <= 0:
            pos_vals = arr[arr > 0]
            return np.min(pos_vals) if pos_vals.size > 0 else 1e-3
        return lower

    # Helper function: get normalization based on chosen scale.
    def get_norm(arr, lower, upper, scale, gamma):
        if scale == 'log':
            lower = safe_lower(arr, lower)
            return LogNorm(vmin=lower, vmax=upper)
        elif scale == 'power':
            return PowerNorm(gamma=gamma, vmin=lower, vmax=upper)
        elif scale == 'sqrt':
            return PowerNorm(gamma=0.5, vmin=lower, vmax=upper)
        elif scale == 'zscale':
            # Use ImageNormalize with a linear stretch based on zscale limits.
            return ImageNormalize(vmin=lower, vmax=upper, stretch=LinearStretch())
        else:  # linear
            return None

    # Determine subplot dimensions.
    n_images = len(data_list)
    if subplot_dims is not None:
        nrows, ncols = subplot_dims
    else:
        nrows = int(np.ceil(n_images / ncols))
    
    # Create subplots with optional shared axes.
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize,
                            sharex=share_axes, sharey=share_axes)
    if n_images == 1:
        axs = [axs]
    else:
        axs = np.array(axs).flatten()
    
    # If sharing vmin/vmax, compute global values.
    if share_vminmax:
        # For zscale, compute global limits using ZScaleInterval
        if scale == 'zscale':
            all_data = np.concatenate([get_array(d).flatten() for d in data_list])
            zscale_interval = ZScaleInterval()
            global_lower, global_upper = zscale_interval.get_limits(all_data)
        else:
            all_data = np.concatenate([get_array(d).flatten() for d in data_list])
            global_lower = np.percentile(all_data, vmin)
            global_upper = np.percentile(all_data, vmax)
    
    # Loop through the datasets and plot each one.
    for i, data in enumerate(data_list):
        arr = get_array(data)  # Assumes a function get_array() that returns a numpy array from the input.
        
        # Determine lower and upper limits.
        if share_vminmax:
            lower, upper = global_lower, global_upper
        else:
            if scale == 'zscale':
                zscale_interval = ZScaleInterval()
                lower, upper = zscale_interval.get_limits(arr)
            else:
                lower = np.percentile(arr, vmin)
                upper = np.percentile(arr, vmax)
        
        # Get normalization (if applicable) for the chosen scale.
        norm = get_norm(arr, lower, upper, scale, gamma)
        
        # Plot the image; if norm is provided, use it; otherwise fall back to vmin/vmax.
        if norm is not None:
            im = axs[i].imshow(arr, cmap=cmap, origin='lower', norm=norm)
        else:
            im = axs[i].imshow(arr, cmap=cmap, origin='lower', vmin=lower, vmax=upper)
        
        axs[i].set_xlabel(xlabel)
        axs[i].set_ylabel(ylabel)
        if titles is not None and i < len(titles):
            axs[i].set_title(titles[i])
        else:
            axs[i].set_title('FITS Plot')
        
        # Add a colorbar for each subplot, shrunk so it isn't too tall.
        cbar = fig.colorbar(im, ax=axs[i], shrink=0.6)
        cbar.set_label(cbar_label)
    
    # Remove any unused subplots.
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])
    
    plt.tight_layout()
    plt.show()
