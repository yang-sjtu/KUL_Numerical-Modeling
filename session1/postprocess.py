import numpy as np
import scipy as sp
import scipy.io as spio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.colors
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy import interpolate

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

def surf_plot(*vargs):#x, y, w, sxx, syy, sxy
    '''
    this function is implemented for post-process of exercise session 1 of Numerical Modeling @ KU Leuven
    '''
# control the post-process with a boolen plot_stress    
    if len(vargs) == 3:
        x = vargs[0]
        y = vargs[1]
        w = vargs[2]
        plot_stress = False
    elif len(vargs) == 6:
        x   = vargs[0]
        y   = vargs[1]
        w   = vargs[2]
        sxx = vargs[3]
        syy = vargs[4]
        sxy = vargs[5]
        plot_stress = True
    else:
        print('input arguments are neither 3 or 6')
    fem = loadmat('fem_result.mat')
# initialization
    NEx  = 50         # aantal elementen in X-richting
    NEy  = 50         #   "        "     "  Y-richting
    NEt  = NEx * NEy  # totaal aantal elementen
    NPx  = NEx + 1    # aantal knooppunten in X-richting
    NPy  = NEy + 1    #   "         "      "  Y-richting
    NPt  = NPx * NPy  # totaal aantal knooppunten
    xE   = np.transpose(np.linspace(0.0, 0.5, NPx))
    yE   = np.transpose(np.linspace(0.0, 0.5, NPy))
    w_FEM = np.empty((NPx,NPy))
    sxx_FEM = np.empty((NPx,NPy))
    syy_FEM = np.empty((NPx,NPy))
    sxy_FEM = np.empty((NPx,NPy))
    svm_FEM = np.empty((NPx,NPy))

# storage of finite elements results in matrices for surf plot  
    for k in range(NPy):
    #index = np.transpose(np.linspace(k*NPx, (k+1)*NPx-1))
        index = range(k*NPx, (k+1)*NPx)
        w_FEM[k,:]   = fem['fem_result'][2].dsp.t3[index] * 1e3
        sxx_FEM[k,:] = fem['fem_result'][2].str.xx[index] / 1e6
        syy_FEM[k,:] = fem['fem_result'][2].str.yy[index] / 1e6
        sxy_FEM[k,:] = fem['fem_result'][2].str.xy[index] / 1e6
        svm_FEM[k,:] = fem['fem_result'][2].str.vm[index] / 1e6
       
    # interpolation of finite elements results on finite differential grid
    w_FEM_F = sp.interpolate.interp2d(xE, yE, w_FEM)
    w_FEM_I = w_FEM_F(x,y)
    sxx_FEM_F = sp.interpolate.interp2d(xE, yE, sxx_FEM)
    sxx_FEM_I = sxx_FEM_F(x,y)
    syy_FEM_F = sp.interpolate.interp2d(xE, yE, syy_FEM)
    syy_FEM_I = syy_FEM_F(x,y)
    sxy_FEM_F = sp.interpolate.interp2d(xE, yE, sxy_FEM)
    sxy_FEM_I = sxy_FEM_F(x,y)
    svm_FEM_F = sp.interpolate.interp2d(xE, yE, svm_FEM)
    svm_FEM_I = svm_FEM_F(x,y)

# transform finite different resutls    
    w = w *1e3
    if plot_stress:
        sxx = sxx / 1e6
        syy = syy / 1e6
        sxy = sxy / 1e6
        svm = np.sqrt(sxx**2 + syy**2 - sxx*syy + 3*sxy**2)

    def color_value(value):
        norm = matplotlib.colors.Normalize(vmin=np.amin(value), vmax=np.amax(value))
        color_value = norm(value)
        return color_value, norm

# calculation of absolute error estimator
    dw_abs   = w - w_FEM_I
    if plot_stress:
        dsxx_abs = sxx - sxx_FEM_I
        dsyy_abs = syy - syy_FEM_I
        dsxy_abs = sxy - sxy_FEM_I
        dsvm_abs = svm - svm_FEM_I

# Make data.
    x, y = np.meshgrid(x, y)

# Plot the surface.
    if not plot_stress:
        plt.figure(3)
        fig = plt.figure(3)
        ax = fig.gca(projection='3d')
    #surf = ax.plot_surface(x, y, w, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    #norm = matplotlib.colors.Normalize(vmin=np.amin(dw_abs), vmax=np.amax(dw_abs))
        color, norm =  color_value(dw_abs)
        surf = ax.plot_surface(x, y, w, facecolors = cm.jet(color) , linewidth=0, antialiased=False)
        m = cm.ScalarMappable(cmap=plt.cm.jet, norm = norm)
        m.set_array([])
        plt.colorbar(m)
    #ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors = cm.jet(Z), linewidth=0, antialiased=False)
    #fig.colorbar(surf, shrink=0.5, aspect=5)
    #ax.set_xlim(0, 0.5)
    #ax.set_ylim(0, 0.5)
        plt.title('absolute deviation of w in mm')
        plt.xlabel('$x$ in m')
        plt.ylabel('$y$ in m')
        ax.set_zlabel('displacement $w$ in mm')
        plt.show()

# visualization of strain
    if plot_stress:
        plt.figure(4)
        fig = plt.figure(4)
        plt.suptitle('Stresses')
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        color, norm =  color_value(sxx)
        surf = ax1.plot_surface(x, y, w, facecolors = cm.jet(color) , linewidth=0, antialiased=False)
        plt.title('contour: $\sigma_{xx}$ in MPa', y=1.16)
        plt.xlabel('$x$ in m')
        plt.ylabel('$y$ in m')
        ax1.set_zlabel('displacement $w$ in mm')
        m = cm.ScalarMappable(cmap=plt.cm.jet, norm = norm)
        m.set_array([])
        plt.colorbar(m)

        ax2 = fig.add_subplot(2, 2, 2, projection='3d')
        color, norm =  color_value(dsxx_abs)
        surf = ax2.plot_surface(x, y, w, facecolors = cm.jet(color) , linewidth=0, antialiased=False)
        plt.title('contour: absolute deviation of $\sigma_{xx}$ compared to FEM', y=1.16)
        plt.xlabel('$x$ in m')
        plt.ylabel('$y$ in m')
        ax2.set_zlabel('displacement $w$ in mm')
        m = cm.ScalarMappable(cmap=plt.cm.jet, norm = norm)
        m.set_array([])
        plt.colorbar(m)
        
        plt.figure(5)
        fig = plt.figure(5)
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        color, norm =  color_value(syy)
        surf = ax1.plot_surface(x, y, w, facecolors = cm.jet(color) , linewidth=0, antialiased=False)
        plt.title('contour: $\sigma_{yy}$ in MPa', y=1.16)
        plt.xlabel('$x$ in m')
        plt.ylabel('$y$ in m')
        ax1.set_zlabel('displacement $w$ in mm')
        m = cm.ScalarMappable(cmap=plt.cm.jet, norm = norm)
        m.set_array([])
        plt.colorbar(m)

        ax2 = fig.add_subplot(2, 2, 2, projection='3d')
        color, norm =  color_value(dsyy_abs)
        surf = ax2.plot_surface(x, y, w, facecolors = cm.jet(color) , linewidth=0, antialiased=False)
        plt.title('contour: absolute deviation of $\sigma_{yy}$ compared to FEM', y=1.16)
        plt.xlabel('$x$ in m')
        plt.ylabel('$y$ in m')
        ax2.set_zlabel('displacement $w$ in mm')
        m = cm.ScalarMappable(cmap=plt.cm.jet, norm = norm)
        m.set_array([])
        plt.colorbar(m)
        
        plt.figure(6)
        fig = plt.figure(6)
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        color, norm =  color_value(sxy)
        surf = ax1.plot_surface(x, y, w, facecolors = cm.jet(color) , linewidth=0, antialiased=False)
        plt.title('contour: $\sigma_{xy}$ in MPa', y=1.16)        
        plt.xlabel('$x$ in m')
        plt.ylabel('$y$ in m')
        ax1.set_zlabel('displacement $w$ in mm')
        m = cm.ScalarMappable(cmap=plt.cm.jet, norm = norm)
        m.set_array([])
        plt.colorbar(m)

        ax2 = fig.add_subplot(2, 2, 2, projection='3d')
        color, norm =  color_value(dsxy_abs)
        surf = ax2.plot_surface(x, y, w, facecolors = cm.jet(color) , linewidth=0, antialiased=False)
        plt.title('contour: absolute deviation of $\sigma_{xy}$ compared to FEM', y=1.16)
        plt.xlabel('$x$ in m')
        plt.ylabel('$y$ in m')
        ax2.set_zlabel('displacement $w$ in mm')
        m = cm.ScalarMappable(cmap=plt.cm.jet, norm = norm)
        m.set_array([])
        plt.colorbar(m)
        
        plt.figure(7)
        fig = plt.figure(7)
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        color, norm =  color_value(svm)
        surf = ax1.plot_surface(x, y, w, facecolors = cm.jet(color) , linewidth=0, antialiased=False)
        plt.title('contour: $\sigma_{vm}$ in MPa', y=1.16)  
        plt.xlabel('$x$ in m')
        plt.ylabel('$y$ in m')
        ax1.set_zlabel('displacement $w$ in mm')
        m = cm.ScalarMappable(cmap=plt.cm.jet, norm = norm)
        m.set_array([])
        plt.colorbar(m)

        ax2 = fig.add_subplot(2, 2, 2, projection='3d')
        color, norm =  color_value(dsvm_abs)
        surf = ax2.plot_surface(x, y, w, facecolors = cm.jet(color) , linewidth=0, antialiased=False)
        plt.title('contour: absolute deviation of $\sigma_{vm}$ compared to FEM', y=1.16) 
        plt.xlabel('$x$ in m')
        plt.ylabel('$y$ in m')
        ax2.set_zlabel('displacement $w$ in mm')
        m = cm.ScalarMappable(cmap=plt.cm.jet, norm = norm)
        m.set_array([])
        plt.colorbar(m)
    plt.show()