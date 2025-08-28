import numpy as np

"""

Readme 

There are 3 different versions of functions to compute the projection correlations

(1) projection_corr(X, Y) allows both X and Y are multivariate,
    X -- n*p 2D array
    Y -- n*p 2D array

(2) projection_corr_1d(X, Y): both X and Y are univariate
    X -- n*1 2D array
    Y -- n*1 2D array

(3) projection_corr_1dy(X, Y): X can be multivariate, Y is univariate
    X -- n*p 2D array
    Y -- n*1 2D array

distance_corr(X, Y) is used to cmpute the distance correlation and the
bias-corrected distance correlation

pearson_corr(X, y) is used to compute the Pearson's correlation

"""


def get_arccos_1d(X):

    # X -- a 1D array
    
    X = np.squeeze(X)
    Y = X[:,None] - X
    Z = Y.T[:,:,None]*Y.T[:,None]
    n = len(X)
    
    a = np.zeros([n, n, n])
    a[Z == 0.] = np.pi/2.
    a[Z < 0.] = np.pi
    
    a = np.transpose(a, (1,2,0))

    a_bar_12 = np.mean(a, axis = 0, keepdims = True)
    a_bar_02 = np.mean(a, axis = 1, keepdims = True)
    a_bar_2  = np.mean(a, axis = (0,1), keepdims = True)
    A = a - a_bar_12 - a_bar_02 + a_bar_2
    
    return a, A


def get_arccos(X):

    # X -- a 2D array
    
    n, p = X.shape
    cos_a = np.zeros([n, n, n])
    
    for r in range(n):
        
        xr = X[r]
        X_r = X - xr
        cross = np.dot(X_r, X_r.T)
        row_norm = np.sqrt(np.sum(X_r**2, axis = 1))
        outer_norm = np.outer(row_norm, row_norm)
        
        zero_idx = (outer_norm == 0.)
        outer_norm[zero_idx] = 1.
        cos_a_kl = cross / outer_norm
        cos_a_kl[zero_idx] = 0.

        cos_a[:,:,r] = cos_a_kl
        
    cos_a[cos_a > 1] = 1.
    cos_a[cos_a < -1] = -1.
    a = np.arccos(cos_a)

    a_bar_12 = np.mean(a, axis = 0, keepdims = True)
    a_bar_02 = np.mean(a, axis = 1, keepdims = True)
    a_bar_2  = np.mean(a, axis = (0,1), keepdims = True)
    A = a - a_bar_12 - a_bar_02 + a_bar_2
        
    return a, A



def pearson_corr(X, y):

    """
    compute Pearson's correlation

    X -- n * p 2D array
    y -- n * 1 2D array or n-dimensional 1D array

    return: n-dimensional 1D array of Pearson's correlation
    """
    
    n, p = X.shape
    y = y.reshape(n, 1)
    
    yc = y - np.mean(y, axis = 0)
    Xc = X - np.mean(X, axis = 0, keepdims = True)
    y_std = yc / np.linalg.norm(yc)
    X_std = Xc / np.linalg.norm(Xc, axis = 0, keepdims = True)
    ps_corr = np.squeeze(np.dot(y_std.T, X_std))
    
    return ps_corr




        
def projection_corr(X, Y):

    """
    compute the projection correlation where
    X -- n*p 2D array
    Y -- n*p 2D array
    """
    
    nx, p = X.shape
    ny, q = Y.shape
    
    if nx == ny:
        n = nx
    else:
        raise ValueError("sample sizes do not match.")
        
    a_x, A_x = get_arccos(X)
    a_y, A_y = get_arccos(Y)
    
    S_xy = np.sum(A_x * A_y) / (n**3)
    S_xx = np.sum(A_x**2) / (n**3)
    S_yy = np.sum(A_y**2) / (n**3)
    
    if S_xx * S_yy == 0.:
        corr = 0.
    else:
        corr = np.sqrt( S_xy / np.sqrt(S_xx * S_yy) )
    
    return corr



def projection_corr_1d(X, Y):

    """
    compute the projection correlation where
    X -- n*1 2D array
    Y -- n*1 2D array
    """
    
    nx, p = X.shape
    ny, q = Y.shape
    
    if nx == ny:
        n = nx
    else:
        raise ValueError("sample sizes do not match.")
        
    a_x, A_x = get_arccos_1d(X)
    a_y, A_y = get_arccos_1d(Y)
    
    S_xy = np.sum(A_x * A_y) / (n**3)
    S_xx = np.sum(A_x**2) / (n**3)
    S_yy = np.sum(A_y**2) / (n**3)
    
    if S_xx * S_yy == 0.:
        corr = 0.
    else:
        corr = np.sqrt( S_xy / np.sqrt(S_xx * S_yy) )
    
    return corr


def projection_corr_1dy(X, Y):

    """
    compute the projection correlation where
    X -- an n*p 2D array
    Y -- an n*1 2D array
    """
    
    nx, p = X.shape
    ny, q = Y.shape
    
    if nx == ny:
        n = nx
    else:
        raise ValueError("sample sizes do not match.")
        
    a_x, A_x = get_arccos(X)
    a_y, A_y = get_arccos_1d(Y)
    
    S_xy = np.sum(A_x * A_y) / (n**3)
    S_xx = np.sum(A_x**2) / (n**3)
    S_yy = np.sum(A_y**2) / (n**3)
    
    if S_xx * S_yy == 0.:
        corr = 0.
    else:
        corr = np.sqrt( S_xy / np.sqrt(S_xx * S_yy) )
    
    return corr


def distance_corr(X, Y):

    """
    compute the distance correlation where
    X -- an n*p 2D array
    Y -- an n*p 2D array

    return: a list of two elements: 
            [distance correlation, bias-corrected distance correlation]
    """
    
    nx, p = X.shape
    ny, q = Y.shape
    
    if nx == ny:
        n = nx
    else:
        raise ValueError("sample sizes do not match.")
        
    if n < 4:
        raise ValueError("sample size is less than 4.")
        
    outer_diff_x = X[:, np.newaxis] - X
    outer_diff_y = Y[:, np.newaxis] - Y
    
    a = np.linalg.norm(outer_diff_x, axis = 2)
    b = np.linalg.norm(outer_diff_y, axis = 2)
    
    a0_bar = np.mean(a, axis = 0, keepdims = True)
    a1_bar = np.mean(a, axis = 1, keepdims = True)
    a_bar  = np.mean(a, axis = (0,1), keepdims = True)
    b0_bar = np.mean(b, axis = 0, keepdims = True)
    b1_bar = np.mean(b, axis = 1, keepdims = True)
    b_bar  = np.mean(b, axis = (0,1), keepdims = True)
    
    A = a - a0_bar - a1_bar + a_bar
    B = b - b0_bar - b1_bar + b_bar
    
    S_xy = np.sum(A*B)
    S_xx = np.sum(A**2)
    S_yy = np.sum(B**2)
    
    if S_xy * S_xx == 0.:
        corr1 = 0.
    else:
        corr1 = np.sqrt(S_xy / np.sqrt(S_xx * S_yy))
        
    A_tilde = a - n*a0_bar/(n-2.) - n*a1_bar/(n-2.) + n*n*a_bar/((n-1.)*(n-2.))
    B_tilde = b - n*b0_bar/(n-2.) - n*b1_bar/(n-2.) + n*n*b_bar/((n-1.)*(n-2.))
    np.fill_diagonal(A_tilde, 0.)
    np.fill_diagonal(B_tilde, 0.)
    
    S_xy_tilde = np.sum(A_tilde*B_tilde)
    S_xx_tilde = np.sum(A_tilde**2)
    S_yy_tilde = np.sum(B_tilde**2)
    
    if S_xy_tilde * S_xx_tilde == 0.:
        corr3 = 0.
    else:
        corr3 = S_xy_tilde / np.sqrt(S_xx_tilde * S_yy_tilde)
    
    return [corr1, corr3]



