'''
Created on 24.07.2018
@author: Stromer, Daniel
'''
import numpy as np
from itertools import combinations_with_replacement
from scipy import ndimage as ndi
from warnings import warn
from skimage import img_as_float

def _image_orthogonal_matrix22_eigvals(M00, M01, M11):
    l1 = (M00 + M11) / 2 + np.sqrt(4 * M01 ** 2 + (M00 - M11) ** 2) / 2
    l2 = (M00 + M11) / 2 - np.sqrt(4 * M01 ** 2 + (M00 - M11) ** 2) / 2
    return l1, l2

def _hessian_matrix_image(H_elems):
    """Convert the upper-diagonal elements of the Hessian matrix to a matrix.
    Parameters
    ----------
    H_elems : list of array
        The upper-diagonal elements of the Hessian matrix, as returned by
        `hessian_matrix`.
    Returns
    -------
    hessian_image : array
        An array of shape ``(M, N[, ...], image.ndim, image.ndim)``,
        containing the Hessian matrix corresponding to each coordinate.
    """
    image = H_elems[0]
    hessian_image = np.zeros(image.shape + (image.ndim, image.ndim))
    for idx, (row, col) in \
            enumerate(combinations_with_replacement(range(image.ndim), 2)):
        hessian_image[..., row, col] = H_elems[idx]
        hessian_image[..., col, row] = H_elems[idx]
    return hessian_image

def hessian_matrix_eigvals(H_elems, Hxy=None, Hyy=None, Hxx=None):
    """Compute Eigenvalues of Hessian matrix.
    Parameters
    ----------
    H_elems : list of ndarray
        The upper-diagonal elements of the Hessian matrix, as returned
        by `hessian_matrix`.
    Hxy : ndarray, deprecated
        Element of the Hessian matrix for each pixel in the input image.
    Hyy : ndarray, deprecated
        Element of the Hessian matrix for each pixel in the input image.
    Hxx : ndarray, deprecated
        Element of the Hessian matrix for each pixel in the input image.
    Returns
    -------
    eigs : ndarray
        The eigenvalues of the Hessian matrix, in decreasing order. The
        eigenvalues are the leading dimension. That is, ``eigs[i, j, k]``
        contains the ith-largest eigenvalue at position (j, k).
    Examples
    --------
    >>> from skimage.feature import hessian_matrix, hessian_matrix_eigvals
    >>> square = np.zeros((5, 5))
    >>> square[2, 2] = 4
    >>> H_elems = hessian_matrix(square, sigma=0.1, order='rc')
    >>> hessian_matrix_eigvals(H_elems)[0]
    array([[ 0.,  0.,  2.,  0.,  0.],
           [ 0.,  1.,  0.,  1.,  0.],
           [ 2.,  0., -2.,  0.,  2.],
           [ 0.,  1.,  0.,  1.,  0.],
           [ 0.,  0.,  2.,  0.,  0.]])
    """
    if Hxy is not None:
        if Hxx is None:
            Hxx = H_elems
        H_elems = [Hxx, Hxy, Hyy]
        warn('The API of `hessian_matrix_eigvals` has changed. Use a list of '
             'elements instead of separate arguments. The old version of the '
             'API will be removed in version 0.16.')
    if len(H_elems) == 3:  # Use fast Cython code for 2D
        eigvals = np.array(_image_orthogonal_matrix22_eigvals(*H_elems))
    else:
        matrices = _hessian_matrix_image(H_elems)
        # eigvalsh returns eigenvalues in increasing order. We want decreasing
        eigvals = np.linalg.eigvalsh(matrices)[..., ::-1]
        leading_axes = tuple(range(eigvals.ndim - 1))
        eigvals = np.transpose(eigvals, (eigvals.ndim - 1,) + leading_axes)
    return eigvals


def hessian_matrix(image, sigma=(1,1), mode='constant', cval=0, order=None):
    """Compute Hessian matrix.
    The Hessian matrix is defined as::
        H = [Hrr Hrc]
            [Hrc Hcc]
    which is computed by convolving the image with the second derivatives
    of the Gaussian kernel in the respective x- and y-directions.
    Parameters
    ----------
    image : ndarray
        Input image.
    sigma : float
        Standard deviation used for the Gaussian kernel, which is used as
        weighting function for the auto-correlation matrix.
    mode : {'constant', 'reflect', 'wrap', 'nearest', 'mirror'}, optional
        How to handle values outside the image borders.
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.
    order : {'xy', 'rc'}, optional
        This parameter allows for the use of reverse or forward order of
        the image axes in gradient computation. 'xy' indicates the usage
        of the last axis initially (Hxx, Hxy, Hyy), whilst 'rc' indicates
        the use of the first axis initially (Hrr, Hrc, Hcc).
    Returns
    -------
    Hrr : ndarray
        Element of the Hessian matrix for each pixel in the input image.
    Hrc : ndarray
        Element of the Hessian matrix for each pixel in the input image.
    Hcc : ndarray
        Element of the Hessian matrix for each pixel in the input image.
    Examples
    --------
    >>> from skimage.feature import hessian_matrix
    >>> square = np.zeros((5, 5))
    >>> square[2, 2] = 4
    >>> Hrr, Hrc, Hcc = hessian_matrix(square, sigma=0.1, order = 'rc')
    >>> Hrc
    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  1.,  0., -1.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0., -1.,  0.,  1.,  0.],
           [ 0.,  0.,  0.,  0.,  0.]])
    """

    image = img_as_float(image)

    gaussian_filtered = ndi.gaussian_filter(image, sigma=(sigma),
                                            mode=mode, cval=cval)

    if order is None:
        if image.ndim == 2:
            # The legacy 2D code followed (x, y) convention, so we swap the axis
            # order to maintain compatibility with old code
            warn('deprecation warning: the default order of the hessian matrix values '
                 'will be "row-column" instead of "xy" starting in skimage version 0.15. '
                 'Use order="rc" or order="xy" to set this explicitly')
            order = 'xy'
        else:
            order = 'rc'


    gradients = np.gradient(gaussian_filtered)
    axes = range(image.ndim)

    if order == 'rc':
        axes = reversed(axes)

    H_elems = [np.gradient(gradients[ax0], axis=ax1)
               for ax0, ax1 in combinations_with_replacement(axes, 2)]

    return H_elems

def _frangi_hessian_common_filter(image, sigma_x, sigma_y,
                                  beta1, beta2):
    """This is an intermediate function for Frangi and Hessian filters.
    Shares the common code for Frangi and Hessian functions.
    Parameters
    ----------
    image : (N, M) ndarray
        Array with input image data.
    scale_range : 2-tuple of floats, optional
        The range of sigmas used.
    scale_step : float, optional
        Step size between sigmas.
    beta1 : float, optional
        Frangi correction constant that adjusts the filter's
        sensitivity to deviation from a blob-like structure.
    beta2 : float, optional
        Frangi correction constant that adjusts the filter's
        sensitivity to areas of high variance/texture/structure.
    Returns
    -------
    filtered_list : list
        List of pre-filtered images.
    """

    if np.any(np.asarray(sigma_x or sigma_y) < 0.0):
        raise ValueError("Sigma values less than zero are not valid")

    beta1 = 2 * beta1 ** 2
    beta2 = 2 * beta2 ** 2

    filtered_array = np.zeros(image.shape)
    lambdas_array = np.zeros(image.shape)

    # Filtering for all sigmas
    # Make 2D hessian
    D = hessian_matrix(image, (sigma_x,sigma_y), order='rc')

    # Correct for scale
    D = np.array(D) * (sigma_x * sigma_y)

    # Calculate (abs sorted) eigenvalues and vectors
    lambda1, lambda2 = hessian_matrix_eigvals(D)

    # Compute some similarity measures
    lambda1[lambda1 == 0] = 1e-10
    rb = (lambda2 / lambda1) ** 2
    s2 = lambda1 ** 2 + lambda2 ** 2

    # Compute the output image
    filtered = np.exp(-rb / beta1) * (np.ones(np.shape(image)) -
                                          np.exp(-s2 / beta2))

    # Store the results in 3D matrices
    filtered_array = filtered
    lambdas_array = lambda1
    return filtered_array, lambdas_array

def frangi(image, sigma_x=1, sigma_y=1, beta1=0.5, beta2=0.05,
           black_ridges=True):
    """Filter an image with the Frangi filter.
    This filter can be used to detect continuous edges, e.g. vessels,
    wrinkles, rivers. It can be used to calculate the fraction of the
    whole image containing such objects.
    Calculates the eigenvectors of the Hessian to compute the similarity of
    an image region to vessels, according to the method described in [1]_.
    
    The difference between the original Frangi filter and this version is
    that different sigmas (sigma_x, sigma_y) for directions can be selected.
    
    Parameters
    ----------
    image : (N, M) ndarray
        Array with input image data.
    sigma_x : float, optional
        Sigma for Gaussian filtering in x-direction (horizontal)
    sigma_y : float, optional
        Sigma for Gaussian filtering in x-direction (vertical)
    beta1 : float, optional
        Frangi correction constant that adjusts the filter's
        sensitivity to deviation from a blob-like structure.
    beta2 : float, optional
        Frangi correction constant that adjusts the filter's
        sensitivity to areas of high variance/texture/structure.
    black_ridges : boolean, optional
        When True (the default), the filter detects black ridges; when
        False, it detects white ridges.
    Returns
    -------
    out : (N, M) ndarray
        Filtered image (maximum of pixels across all scales).
    Notes
    -----
    Written by Marc Schrijver, 2/11/2001
    Re-Written by D. J. Kroon University of Twente (May 2009)
    References
    ----------
    .. [1] A. Frangi, W. Niessen, K. Vincken, and M. Viergever. "Multiscale
           vessel enhancement filtering," In LNCS, vol. 1496, pages 130-137,
           Germany, 1998. Springer-Verlag.
    .. [2] Kroon, D.J.: Hessian based Frangi vesselness filter.
    .. [3] http://mplab.ucsd.edu/tutorials/gabor.pdf.
    """
    filtered, lambdas = _frangi_hessian_common_filter(image,
                                                      sigma_x, sigma_y,
                                                      beta1, beta2)
    if black_ridges:
        filtered[lambdas < 0] = 0
    else:
        filtered[lambdas > 0] = 0

    # Return for every pixel the value of the scale(sigma) with the maximum
    # output pixel value
    return filtered