#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for dimers on the 2x1 weighted triangular lattice
"""

import numpy as np
import time
import scipy.sparse as sps
import scipy as sp

import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection

cycle = plt.rcParams['axes.prop_cycle'].by_key()['color'] # default colors (length 10)

def setfont():
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Times"
        })
    #plt.rcParams['font.serif'] = ['Times']
    return 0


#%% Bools

def square_bool(n,i,j):
    '''square shape'''
    if i in range(n) and j in range(n):
        return True
    return False

def aztec_bool(n,i,j):
    '''Aztec diamond'''
    if i in range(n) and j in range(n):
        if n//2-1 <= i+j <= 3*n//2-1 and j-n//2 <= i <= j+n//2:
            return True
    return False

#%% Coordinate functions

def matcoords(n, shape_bool):
    '''return dictionary of entries lattice-coords (i,j): matrixcoord'''
    coordmap = {}
    mcoord = 0
    width, height = xy_dimensions(n, shape_bool)
    for i in range(height):
        for j in range(width):
            if shape_bool(n,i,j): # if (i,j) in shape
                coordmap[(i,j)] = mcoord
                mcoord += 1
    return coordmap

def num_vertices(n, shape_bool):
    '''
    Returns number of vertices in the region defined by 'shape_bool'
    '''
    return len(matcoords(n,shape_bool))

def shape_coord_from_bool(n, shape_bool):
    '''
    Returns a shape_coord function
    '''
    coordmap = matcoords(n, shape_bool)
    return lambda n, i, j: coordmap[(i,j)]


def is_in_shape(n,i,j, shape_coord):
    '''
    Return True if lattice coordinate (i,j) is in the shape described by 
    'shape_coord', False if not.
    '''
    try:
        shape_coord(n,i,j)
        return True
    except:
        return False

#%% Kasteleyn matrix and lattice paths

def kasteleyn_tri21(n, shape_bool, weightlist=[1]*6, dtype=float):
    '''
    Create the Kasteleyn matrix for the triangular lattice with 2x1 periodic
    edge weights given by 'weightlist', where the edge order is
    [horizontal1, horizontal2, vertical1, vertical2, diagonal1, diagonal2]
    
    Takes shape_bool not shape_coord
    '''
    # if shape_coord.__name__ not in ['square_coord', 'aztec_coord']:
    #     return '%s not supported'%shape_coord.__name__
        
    shape_coord = shape_coord_from_bool(n, shape_bool)
    # the number of vertices in the lattices
    size = num_vertices(n, shape_bool)
    width = n if shape_bool.__name__ != 'stretch_aztec_bool' else 2*n
    
    a0,a1,b0,b1,c0,c1 = tuple(weightlist)
    hweights = [a0,a1]
    vweights = [b0,b1]
    dweights = [c0,c1]
    
    mat = sps.lil_array((size,size),dtype=dtype)
    
    # loop over lattice coordinates
    for i in range(n):
        for j in range(width):
            if is_in_shape(n,i,j,shape_coord) == True:
                # get the matrix coordinate
                coord = shape_coord(n,i,j)
                
                # check if its neighbors are in the shape too
                # the i,j coordinates are REVERSED (i = vertical movement, j = horizontal)
                # direction coordinates are usual xy, e.g. (1,0) = right
                for direction in [(1,0),(0,1),(1,1)]:
                    neighbor = i + direction[1], j + direction[0]
                    if is_in_shape(n,neighbor[0],neighbor[1],shape_coord) == True:
                        # Determine the sign and weighting for the edge
                        match direction:
                            case (1,0):
                                weight = 1 * hweights[j%2]
                            case (0,1):
                                weight = (-1)**j * vweights[j%2]
                            case (1,1):
                                weight = (-1)**(j+1) * dweights[j%2]
                        
                        # Add the neighbor
                        newcoord = shape_coord(n,neighbor[0],neighbor[1])
                        if direction != (1,1) or \
                            (direction == (1,1) and is_in_shape(n,i+1,j, shape_coord)\
                             and is_in_shape(n,i,j+1,shape_coord)): 
                            mat[coord, newcoord] = weight
                            mat[newcoord, coord] = -weight
                                                        
    return mat.tocsc()


def triangle_string(n, pathlength, vert=True, colshift=0):
    '''
    Return vertical (top->down) string from center if vert=True
    (adjust column shift with 'colshift')
    
    Return horizontal (center->left) string if vert=False
    (adjust row shift with 'colshift')
    
    'pathlength' is the length in lattice coordinates, not in edge/face distance
    '''
    string = []

    if vert: # vertical path
        startcol = n//2-1 + colshift
        end = pathlength-1 # because it is convenient to put pathlength=n//2
        for l in range(1,end):
            string.append([(n//2-l, startcol), (n//2-l, startcol+1)])
            string.append([(n//2-l, startcol+1), (n//2-l-1, startcol)]) # diagonal edge
        string.append([(n//2-end, startcol), (n//2-end, startcol+1)])
        
    else: # horizontal path is the same but swap x-y coordinates
        startrow = n//2-1 + colshift
        end = pathlength-1 # because it is convenient to put pathlength=n//2
        for l in range(1,end):
            string.append([(startrow, n//2-l), (startrow+1, n//2-l)])
            string.append([(startrow+1, n//2-l), (startrow, n//2-l-1)]) # diagonal edge
        string.append([(startrow, n//2-end), (startrow+1, n//2-end)])
        
    return string


#%% vison and dimer functions

def kinv_lu_sparse_large(n, kmatsparse, shape_coord, matcoords, verbose=True):
    ''' 
    Return the small K^{-1} matrix according to the matrix indices in 'matcoords'
    using sparse LU floating point method
    '''
    ksize = np.shape(kmatsparse)[0]
    
    Asize = len(matcoords) 
    
    t0 = time.time()
    lusolver = sps.linalg.splu(kmatsparse)
    t1 = time.time()
    if verbose:
        print("Computed sparse LU factorization in %.2f seconds"% (t1-t0))
    
    kinvsmall = np.zeros((Asize,Asize),dtype=float)
    for index in range(Asize):
        standard_basis_v = np.transpose(np.eye(1, ksize, matcoords[index]))
        kinvsmall[:,index] = (lusolver.solve(standard_basis_v))[matcoords,0]
    
    return kinvsmall


def vison_large(n, ksmall, kinvsmall, length=''):
    '''
    Calculate visons along a zigzag path in the triangular lattice.
    
    The vertex order for 'ksmall' and 'kinvsmall' must be the order of vertices
    in the zigzag path (viewing the zigzag shape itself as the 'path' here)
    
    Note: This actually calculates vison correlator at face distance 'length-1'
    '''
    ksmallsize = np.shape(ksmall)[0]
    if length == '':
        size = ksmallsize
    else:
        size = length
    
    # Create ksmalledges, which only has nonzero entries for edges in the path
    # note this is not the diagonal for non-bipartite indexing
    kse = ksmalledges(n, ksmall)
    
    val = np.linalg.det(np.eye(size) - 2*kinvsmall[:size,:size] @ kse[:size,:size])
    
    # optional check
    assert val >-10**-15, 'Needs to be >0, received %s'%str(val)
    
    return np.sqrt(np.abs(val))

def ksmalledges(n, ksmall, dtype=float):
    '''
    Return small matrix K|_{path} used in calculating the vison correlator.
    It is a small matrix K but only edges in the path are represented as nonzero.
    '''
    size = np.shape(ksmall)[0]
    ksmalledges = np.zeros((size,size), dtype=dtype)
    for vindex in range(size-1):
        ## connect to neighbor
        ksmalledges[vindex, vindex+1] = ksmall[vindex, vindex+1]
        ksmalledges[vindex+1, vindex] = ksmall[vindex+1, vindex]
    return ksmalledges


def vd_triangular(n, shape_bool, pathlength=10,weightlist=[1]*6, xlog=False,\
                    verbose=True, plot=True, vert=True, colshift=0):
    '''
    Return vison and dimer correlators for 2x1 periodic triangular lattice
    along string 'triangle_string(n, pathlength, vert, colshift)'
    
    pathlength is in lattice coordinates
    '''
    kmat = kasteleyn_tri21(n, shape_bool, weightlist)

    string = triangle_string(n,pathlength, vert=vert, colshift=colshift)
    
    shape_coord = shape_coord_from_bool(n, shape_bool)
    # generate matrix coordinates for all vertices in the path
    matcoords = []
    for index in range(len(string)):
        v0, v1 = string[index]
        
        # For zigzag path, only add the first vertex of the edge because the
        # 2nd vertex will be added with the next edge. 
        # Except for the last edge, in that case add the 2nd vertex too.
        matcoords.append(shape_coord(n, v0[0], v0[1]))
        if index == len(string)-1:
            matcoords.append(shape_coord(n, v1[0], v1[1]))
    if verbose:
        print(matcoords, len(matcoords))
    
    kinvs = kinv_lu_sparse_large(n, kmat, shape_coord, matcoords=matcoords, \
                           verbose=False)

    ksmall = kmat[np.ix_(matcoords, matcoords)].toarray()

    v = [vison_large(n, ksmall, kinvs, length=l) for l in range(1,len(string))]
    # as noted in 'vison_large', length=l is actually the vison correlator at l-1
    # so v starts from face distance = 0
    if plot:
        plot_points(v,xlog=xlog, title='visons')
    
    # also compute dimer correlations
    d = []
    for i in range(2, len(string)):
        corr = dimer_large(ksmall, kinvs, i)
        d.append(corr)
    if plot:
        plot_points(d, xlog=xlog, title='dimers')
    
    return v, d

def dimer_large(ksmall, kinvs, distance):
    '''
    dimer-dimer correlator for the zigzag path on triangular lattice
    '''
    assert distance >= 2, 'distance must be \ge 2 for zigzag path'
    dboth = -kinvs[0,distance]*kinvs[1,distance+1]+kinvs[0,distance+1]*kinvs[1,distance]    
    return np.abs(dboth * ksmall[0,1] * ksmall[distance,distance+1])


def kinv_corr(n, shape_bool, pathlength=10,weightlist=[1]*6, xlog=False,\
                    verbose=True, plot=True, vert=True, colshift=0):
    '''
    Return K^{-1}(v0, vl) values for 2x1 periodic triangular lattice
    along string 'triangle_string(n, pathlength, vert, colshift)'
    '''
    kmat = kasteleyn_tri21(n, shape_bool, weightlist)

    string = triangle_string(n,pathlength, vert=vert, colshift=colshift)
    
    shape_coord = shape_coord_from_bool(n, shape_bool)
    # generate matrix coordinates for all vertices in the path
    matcoords = []
    for index in range(len(string)):
        v0, v1 = string[index]
        
        # For zigzag path, only add the first vertex of the edge because the
        # 2nd vertex will be added with the next edge. 
        # Except for the last edge, in that case add the 2nd vertex too.
        matcoords.append(shape_coord(n, v0[0], v0[1]))
        if index == len(string)-1:
            matcoords.append(shape_coord(n, v1[0], v1[1]))
    if verbose:
        print(matcoords, len(matcoords))
    
    kinvs = kinv_lu_sparse_large(n, kmat, shape_coord, matcoords=matcoords, \
                           verbose=False)
        
    kicorr = kinvs[0,1:] #[np.abs(kinvs[0,l]) for l in range(1,len(string))]
    # as noted in 'vison_large', length=l is actually the vison correlator at l-1
    # so v starts from face distance = 0
    if plot:
        plot_points(kicorr,xlog=xlog, title=r'$|K^{-1}(v_0,v_1)|$')
    
    return kicorr


#%% Save and load visons and dimers

'''
Some useful lists of a values

alistfull=np.concatenate((np.arange(0,4.1,.1),[4.25,4.5,4.75,5,5.5,6], [2.85,2.95,3.01,3.02,3.05,3.15]))
alist2=np.concatenate((np.arange(0,4.1,.1),[4.25,4.5,4.75,5,5.5,6]))

# only between 2 to 4
alist24=np.concatenate((np.arange(2,4.1,.1), [2.85,2.95,3.01,3.02,3.05,3.15]))
alist242=np.arange(2,4.1,.1)
'''


def save_vd(n, alist= [0.5,1,2.5,3,3.5,4,5], vert=True, colshift=0, \
            filepath='./', verbose=True):
    '''
    Calculate and save vison and dimer correlators for all a values in 'alist',
    along path specified by 'triangle_string(n,n//2,vert,colshift)'
    
    INPUT
        n: system size
        alist: list of a values to run
        vert [optional]: If True, uses a vertical path. If False, uses horizontal path.
        colshift [optional]: Amount to shift the path by
        filepath [optional]: where to save npy files (relative file path)
        verbose [optional]: If True, prints progress when it finishes an a value.
    '''
    if filepath[-1] != '/':
        filepath += '/'
        
    vstring = 'v' if vert else 'h'
    for a in alist:
        v, d = vd_triangular(n,square_bool,n//2,[a,1,1,1,1,1],verbose=False,\
                             plot=False,vert=vert,colshift=colshift)
        np.save(filepath + 'tri_visons_n%i_a%.2f_%s_shift%i.npy'%(n,a,vstring,colshift), v)
        np.save(filepath + 'tri_dimers_n%i_a%.2f_%s_shift%i.npy'%(n,a,vstring,colshift), d)
        
        if verbose:
            print('saved visons and dimers for a=%.2f'%a)
    
    return 0

def load_vd(n, alist= [0.5,1,2.5,3,3.5,4,5], vert=True, colshift=0, filepath='./'):
    '''
    Load saved visons and dimers from folder 'filepath'. 
    
    Options specify which files to load, same description as in 'save_vd'.
    Will throw an error if file doesn't exist.
    '''
    if filepath[-1] != '/':
        filepath += '/'
        
    num_a = len(alist)
    vlist = np.empty((num_a, n-4))
    dlist = np.empty((num_a, n-5))
    
    vstring = 'v' if vert else 'h'
    
    for index, a in enumerate(alist):
        vlist[index,:] = np.load(filepath + 'tri_visons_n%i_a%.2f_%s_shift%i.npy'%(n,a,vstring,colshift))
        dlist[index,:] = np.load(filepath + 'tri_dimers_n%i_a%.2f_%s_shift%i.npy'%(n,a,vstring,colshift))
    
    return vlist, dlist


# K^{-1} values - written and run after already saved visons/dimers

def save_ki(n, alist= [0.5,1,2.5,3,3.5,4,5], vert=True, colshift=0, \
            filepath='./', verbose=True):
    '''
    Calculate and save vison and dimer correlators for all a values in 'alist',
    along path specified by 'triangle_string(n,n//2,vert,colshift)'
    
    INPUT
        n: system size
        alist: list of a values to run
        vert [optional]: If True, uses a vertical path. If False, uses horizontal path.
        colshift [optional]: Amount to shift the path by
        filepath [optional]: where to save npy files (relative file path)
        verbose [optional]: If True, prints progress when it finishes an a value.
    '''
    if filepath[-1] != '/':
        filepath += '/'
        
    vstring = 'v' if vert else 'h'
    for a in alist:
        ki = kinv_corr(n,square_bool,n//2,[a,1,1,1,1,1],vert=vert,\
                       colshift=colshift,verbose=False, plot=False)
        np.save(filepath + 'tri_kinv_n%i_a%.2f_%s_shift%i.npy'%(n,a,vstring,colshift), ki)
        
        if verbose:
            print('saved K^{-1} along string for a=%.2f'%a)
    
    return 0

def load_ki(n, alist= [0.5,1,2.5,3,3.5,4,5], vert=True, colshift=0, filepath='./'):
    '''
    Load saved K^{-1} values from folder 'filepath'. 
    
    Options specify which files to load, same description as in 'save_ki'.
    Will throw an error if file doesn't exist.
    '''
    if filepath[-1] != '/':
        filepath += '/'
        
    num_a = len(alist)
    kilist = np.empty((num_a, n-3))
    
    vstring = 'v' if vert else 'h'
    
    for index, a in enumerate(alist):
        kilist[index,:] = np.load(filepath + 'tri_kinv_n%i_a%.2f_%s_shift%i.npy'%(n,a,vstring,colshift))
    
    return kilist




#%% vison and dimer plots, correlation lengths

def dimerplot_slopes(dlist, alist, pathlength, ymin=0, pt=14, savename='', \
                     plotslopes=False, startslope=20, end=100, verbose=False, \
                         showlegend=True,matchcolor=True, ymax=1,legendloc='best',\
                             plot=True, colorlist=cycle,text='',xtext=0,ytext=.1):
    '''
    Plots dimer-dimer correlator and returns best fit inverse correlation length
    (= -slope after taking a logarithm)

    Parameters
    ----------
    vlist : np.array. 
        The different rows correspond to different values of the parameter a
        in 'alist'.
        Row i should be the vison correlator along the path, for parameter 
        value alist[i].
        
    alist : list of a values correponding to 'vlist'
    pathlength : int, how many points to plot
    
    [optional parameters] See 'visonplot_slopes' description.
    '''
    num_a = len(alist)
    
    # get inverse correlation lengths
    slopelist = []
    interceptlist = []

    for i in range(num_a):
        regvals = sp.stats.linregress(range(startslope, end+1), \
                                np.log(np.abs(np.array(dlist[i,:][startslope-2:end+1-2]))))
        if verbose:
            print(regvals)
            
        slope, intercept = regvals[:2]
        slopelist.append(slope)
        interceptlist.append(intercept)
    
    if plot:
        markerlist = ["^","s","d", "x","o","v","*"]
        for i in range(num_a):
            plt.plot(range(2,pathlength+2), dlist[i,:pathlength],label=r'$\alpha=%.2f$'%alist[i],\
                     marker=markerlist[i%len(markerlist)], linestyle='',fillstyle='none',\
                         color=colorlist[i%len(colorlist)])
        
        if plotslopes: # plot best fit slope
            xvals = np.arange(startslope,pathlength+2)
            slopecolor = colorlist if matchcolor else ['k']
            for i in range(num_a):
                plt.plot(xvals, np.exp(slopelist[i]*xvals+interceptlist[i]), \
                         color=slopecolor[i%len(slopecolor)])
        
        plt.yscale('log')
        if showlegend:
            plt.legend(fontsize=pt-2,labelspacing=0.2, handletextpad=0.1, \
                       borderpad=.2, handlelength=1, loc=legendloc)
        plt.xlabel(r'Edge distance $\ell$',fontsize=pt)
        plt.ylabel(r'$|\langle d_0d_\ell\rangle-\langle d_0\rangle\langle d_\ell\rangle|$',\
                   fontsize=pt)
        plt.xticks(fontsize=pt-2)
        plt.yticks(fontsize=pt-2)
        #plt.title('Dimer-dimer correlator',fontsize=pt)
        if ymin > 0:
            plt.ylim(bottom=ymin)
        if ymax < 1:
            plt.ylim(top=ymax)
        
        if len(text) > 0:
            plt.text(xtext,ytext,text,fontsize=pt)
        
        if savename != '':
            plt.savefig(savename, bbox_inches='tight')
        
        plt.show()
    return slopelist


def visonplot_slopes(vlist, alist, pathlength, ymin=0, pt=14, savename='',\
                     ylog=True,xlog=False,text='',xtext=0,ytext=0,showlegend=True,\
                        plotslopes=False, startslope=20, end=100, verbose=False,\
                            matchcolor=True, threshold=10**-15,ymax=1, \
                                startcolorindex=0,legendloc='best', plot=True,\
                                    quiet=False,colorlist=cycle):
    '''
    Plots vison correlators and returns best fit inverse correlation length
    (= -slope after taking a logarithm)

    Parameters
    ----------
    vlist : np.array. 
        The different rows correspond to different values of the parameter a
        in 'alist'.
        Row i should be the vison correlator along the path, for parameter 
        value alist[i].
        
    alist : list of a values correponding to 'vlist'
    pathlength : int, how many points to plot
    
    [Optional arguments]
    ymin : float, minimum value for y-axis in plot. If = 0, uses default instead.
    pt : fontsize. int
    savename : name to save figure to
    ylog : If True, makes y-axis logarithmic scale
    xlog : If True, makes x-axis logarithmic scale
    text : Text to place on figure. Useful for labels (a), (b), etc.
    xtext : text x-coordinate placement (wrt coordinates used in plot)
    ytext : text y-coordinate placement (wrt coordinates used in plot)
    showlegend : If True, includes plot legend. 
                 Only useful to have this be False when plotting LOTS of values of a
    plotslopes : If True, plots best fit exponential curve for each value of a
    startslope : edge/face distance to start at for the slope regression. The default is 20.
    end : edge/face distance to end the slope regression. The default is 100.
    verbose : If True, prints all slope regression outputs.
    matchcolor : If True, and best fit curves are plotted, uses matching color
                 for the data and the fit curve. Otherwise, curve is plotted in black.
    threshold : Do not do best fit curve fit on any points below this threshold.
    ymax : max y-value on plot. If = 1, uses default instead.
    startcolorindex : Specifies where in the color and markerstyle cycle to start.
                      Useful if we are splitting up different regimes of 
                      the parameter a into different plots.
    legendloc : specify placement of legend
    plot : If True, make plots. Otherwise, this function just returns the best
                                fit slopes.
    quiet : If False, prints when the slope fitting has to truncate due to
                      values below 'threshold'
    colorlist : List of colors to use for plotting

    Returns
    -------
    slopelist : list of best fit slopes, in order of the parameters in 'alist'

    '''
    num_a = len(alist)
    
    # get inverse correlation lengths
    slopelist = []
    interceptlist = []
    
    # calculate slopes
    good_indices_lengths = []
    for i in range(num_a):
                
        data = np.abs(np.array(vlist[i,:][startslope:end+1]))
        good_indices = np.where(data>threshold)
        good_data = data[good_indices[0]]
        gd_length = len(good_indices[0])
        good_indices_lengths.append(gd_length)
        
        if not quiet:
            if gd_length < len(data):
                print('a=%.2f: vison slope fit stopping at face distance %i'%(alist[i],gd_length+startslope))
        
        regvals =  sp.stats.linregress(np.arange(startslope, end+1)[good_indices[0]], \
                                    np.log(good_data))
        if verbose:
            print(regvals)
        slope, intercept =  regvals[:2]
        slopelist.append(slope)
        interceptlist.append(intercept)
        #print('slope for a=%.2f, from l=%i to %i: %s'%(alist[i], startslope,end,slope))
    
    if plot:
        # plot data
        markerlist = ["^","s","d", "x","o","v","*"]
        for i in range(num_a):
            plt.plot(range(1,pathlength), vlist[i,1:pathlength],label=r'$\alpha=%.1f$'%alist[i], \
                     marker=markerlist[(i+startcolorindex)%len(markerlist)],\
                     linestyle='',fillstyle='none', \
                     color=colorlist[(i+startcolorindex)%len(colorlist)])
    
        if plotslopes: # plot best fit line
            slopecolor = colorlist if matchcolor else ['k']
            for i in range(num_a):
                xvals = np.arange(startslope,pathlength)[:good_indices_lengths[i]]
                plt.plot(xvals, np.exp(slopelist[i]*xvals+interceptlist[i]), \
                         color=slopecolor[i%len(slopecolor)])
        
        if ymin > 0:
            plt.ylim(bottom=ymin)
        if ymax != 1:
            plt.ylim(top=ymax)
        
        if ylog:
            plt.yscale('log')
        if xlog:
            plt.xscale('log')
        if showlegend:
            plt.legend(fontsize=pt-2,labelspacing=0.2, handletextpad=0.1,\
                       borderpad=.2, handlelength=1, loc=legendloc)
        plt.xlabel(r'Face distance $\ell$',fontsize=pt)
        plt.ylabel(r'$|\langle v_0v_\ell\rangle|$',fontsize=pt)
        plt.xticks(fontsize=pt-2)
        plt.yticks(fontsize=pt-2)
        #plt.ylim(bottom=ymin)
        
        if len(text) > 0:
            plt.text(xtext,ytext,text,fontsize=pt)
            
        if savename != '':
            plt.savefig(savename, bbox_inches='tight')
        
        plt.show()
    return slopelist

def corr_lengths(n, alist, startslope=30, end=100, pt=18,\
                 savename='',save=False,filepath='./',vert=True,colshift=0,\
                    plot=True, return_data=False):
    '''
    Plots inverse correlation lengths as a function of parameter 'a' in 'alist'
    
    Loads saved .npy dimer and vison correlator data.

    Parameters
    ----------
    n : int. system size.
    alist : list of a values correponding to 'vlist'
    startslope : edge/face distance to start at for the slope regression. The default is 20.
    end : edge/face distance to end the slope regression. The default is 100.
    pt : int. font size.
    savename : name to save figure to. If non-empty, will save even if 'save' is False.
    save : If True, save figure to file. If 'savename' is empty, will use a default name.
    filepath : path to dimer and vison .npy files.
    vert : If True, will try to load files for vertical path. 
           If False, will try to load files for horizontal path
    colshift : lattice path shift amount, for files to load.
    plot : If False, will not plot intermediate dimer-dimer and vison correlator plots.
    return_data : If True, will return the best-fit slopes for dimers and visons
    '''
    plt.rcParams["figure.figsize"] = (6.4,4.2)
    
    alist = np.sort(alist)
    vlist,dlist=load_vd(n,alist=alist,filepath=filepath,vert=vert, colshift=colshift)
    
    dslopes = dimerplot_slopes(dlist, alist, end, startslope=startslope,end=end,\
                               ymin=0,plotslopes=True,showlegend=False,\
                                   plot=plot, matchcolor=False)
    plt.show()
    vslopes = visonplot_slopes(vlist, alist, end, startslope=startslope,end=end,\
                               plotslopes=True,showlegend=False,\
                                   plot=plot, matchcolor=False)
    plt.show()
    
    fig, ax = plt.subplots()

    ax.plot(alist, -np.array(dslopes), color='blue', marker='o',fillstyle='none',label='Dimers')

    ax.set_ylabel('Inverse correlation length', fontsize=pt)
    ax.set_xlabel(r'$\alpha$', fontsize=pt)
    ax.tick_params(labelsize=pt-2)
    ax.tick_params(axis='y')
    
    ax.axvline(x=3,color='k',linestyle='dashed')
    ax.axhline(y=0,color='k',linestyle='dashed')
    
    ax.plot(alist,-np.array(vslopes),color='darkorange', marker='v', fillstyle='none',label='Visons')
        
    ax.legend(fontsize=pt-2,labelspacing=0.2, handletextpad=0.1, borderpad=.2, handlelength=1)
    
    if save==True or savename !='':
        if savename == '':
            savename = 'invcorrelationlength_n%i_%i-%i.pdf'%(n,startslope,end)
        plt.savefig(savename, bbox_inches='tight')
    
    plt.show()
    
    if return_data:
        return dslopes, vslopes
    return 0

def corr_lengths_all(n, list_of_alists, startslope=30, end=100, pt=16,\
                 savename='',save=False,filepath='./',quiet=True,\
                     legendloc='best', ymax=1, return_data=False, figheight=4.2):
    '''
    Plot correlation lengths along multiple paths all together.
    
    list_of_alists order: verrtical shift=0, vertical shift=1, horizontal
    '''
    
    plt.rcParams["figure.figsize"] = (6.4,figheight)
    
    irange = [0,2]
    a_v0, a_v1, a_h = list_of_alists
    
    vlist_v0, dlist_v0 = load_vd(n, np.sort(a_v0), filepath=filepath,vert=True, colshift=0)
    vlist_v1, dlist_v1 = load_vd(n, np.sort(a_v1), filepath=filepath,vert=True, colshift=1)
    vlist_h, dlist_h = load_vd(n, np.sort(a_h), filepath=filepath,vert=False)
    
    vlistlist = [vlist_v0, vlist_v1, vlist_h]
    dlistlist = [dlist_v0, dlist_v1, dlist_h]
    
    fig, ax = plt.subplots()
    
    ax.axvline(x=3,color='k',linestyle='dashed')
    ax.axhline(y=0,color='k',linestyle='dashed')

    # set plot markers and labels
    fillstyle = ['none', 'full', 'none']
    msize = [6,4,6]
    
    cmap = mpl.colormaps['plasma']
    #colors = mpl.colormaps['plasma'](np.linspace(0,1,5))
    
    #dcolors = [colors[0],colors[1],colors[1]]
    dcolors = ['blue', 'dodgerblue', cmap(.3)]
    dmshape = ["o", ".", "s"]
    dlabels = ['Dimers: vertical path', '', 'Dimers: horizontal path']
    
    #vcolors = [colors[2],colors[3],colors[3]]
    vcolors = [cmap(.6), 'tab:pink', 'tab:olive']
    vmshape = ["v", ".", "^"]
    vlabels = ['Visons: vertical path', '', 'Visons: horizontal path']

    
    dslopeslist = []
    vslopeslist = []
    # i=0: vertical, i=1: adjacent vertical, i=2: horizontal
    for i in irange:
        dlist = dlistlist[i]
        vlist = vlistlist[i]
        alist = np.sort(list_of_alists[i])
        
        dslopes = dimerplot_slopes(dlist, alist, end, startslope=startslope,end=end,\
                                   ymin=0,plotslopes=True,showlegend=False,\
                                       matchcolor=False,plot=False)
        vslopes = visonplot_slopes(vlist, alist, end, startslope=startslope,end=end,\
                                   plotslopes=True,showlegend=False,matchcolor=False,\
                                       plot=False,quiet=quiet)
    
        ax.plot(alist, -np.array(dslopes), color=dcolors[i], marker=dmshape[i],\
                fillstyle=fillstyle[i],label=dlabels[i],ms=msize[i])
        ax.plot(alist,-np.array(vslopes),color=vcolors[i], marker=vmshape[i], \
                fillstyle=fillstyle[i],label=vlabels[i],ms=msize[i])
            
        dslopeslist.append(dslopes); vslopeslist.append(vslopes) # useful for return

    ax.set_ylabel('Inverse correlation length', fontsize=pt)
    ax.set_xlabel(r'$\alpha$', fontsize=pt)
    ax.tick_params(labelsize=pt-2)
    ax.tick_params(axis='y')
    if ymax != 1:
        ax.set_ylim(top=ymax)
    
    # reorder the labels in the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [0,2,1,3]
    ax.legend([handles[i] for i in order],\
               [labels[i] for i in order],\
               fontsize=pt-2, labelspacing=0.2, handletextpad=0.1, borderpad=.2,\
                   ncol=1,handlelength=1, loc=legendloc)
        
    if save==True or savename !='':
        if savename == '':
            savename = 'invcorrelationlength_n%i_%i-%i_all.pdf'%(n,startslope,end)
        plt.savefig(savename, bbox_inches='tight')
    
    plt.show()
    
    if return_data:
        return dslopeslist, vslopeslist
    return 0


def corr_lengths_largea(n, alarge=np.concatenate((range(3,20),range(20,52,2))),\
                        startslope=10, end=150, pt=16,\
                 savename='',save=False,filepath='./',quiet=True,\
                     legendloc='best', return_data=False):
    '''
    Plot correlation lengths for large a.
    '''
    
    plt.rcParams["figure.figsize"] = (6.4,4.2)
        
    vlist_v0, dlist_v0 = load_vd(n, np.sort(alarge), filepath=filepath,vert=True, colshift=0)
    vlist_h, dlist_h = load_vd(n, np.sort(alarge), filepath=filepath,vert=False)
    
    vlistlist = [vlist_v0, vlist_h]
    dlistlist = [dlist_v0, dlist_h]
    
    fig, ax = plt.subplots()
    
    ax.axvline(x=3,color='k',linestyle='dashed')
    ax.axhline(y=0,color='k',linestyle='dashed')

    # set plot markers and labels
    fillstyle = ['none', 'none']
    msize = [6,6]
    
    cmap = mpl.colormaps['plasma']
    
    dcolors = ['blue', cmap(.3)]
    dmshape = ["o", "s"]
    dlabels = ['Dimers: vertical path', 'Dimers: horizontal path']
    
    dslopeslist = []
    vslopeslist = []
    
    # plot logarithm
    avals = np.array(alarge)
    ax.plot(avals, 1.05*np.log((avals-1)/2), 'k', label=r'$1.05\log(\frac{\alpha-1}{2})$')
    ax.plot(avals, 0.575*np.log((avals-1)/2), 'k', label=r'$0.575\log(\frac{\alpha-1}{2})$')

    # plot dimer correlator
    for i in [0,1]:
        dlist = dlistlist[i]
        vlist = vlistlist[i]
        
        dslopes = dimerplot_slopes(dlist, alarge, end, startslope=startslope,end=end,\
                                   ymin=0,plotslopes=True,showlegend=False,\
                                       matchcolor=False,plot=False)
        vslopes = visonplot_slopes(vlist, alarge, end, startslope=startslope,end=end,\
                                   plotslopes=True,showlegend=False,matchcolor=False,\
                                       plot=False,quiet=quiet)
    
        ax.plot(alarge, -np.array(dslopes), color=dcolors[i], marker=dmshape[i],\
                fillstyle=fillstyle[i],label=dlabels[i],ms=msize[i],linestyle='')

        dslopeslist.append(dslopes); vslopeslist.append(vslopes) # useful for return


    ax.set_ylabel('Inverse correlation length', fontsize=pt)
    ax.set_xlabel(r'$\alpha$', fontsize=pt)
    ax.tick_params(labelsize=pt-2)
    ax.tick_params(axis='y')
    
    
    
    # reorder the labels in the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [2,3,0,1] #[0,2,1,3,4,5]
    ax.legend([handles[i] for i in order],\
               [labels[i] for i in order],\
               fontsize=pt-2, labelspacing=0.2, handletextpad=0.5, borderpad=.2,\
                   ncol=1,handlelength=1, loc=legendloc)
        
    if save==True or savename !='':
        if savename == '':
            savename = 'invcorrelationlength_n%i_%i-%i_all.pdf'%(n,startslope,end)
        plt.savefig(savename, bbox_inches='tight')
    
    plt.show()
    
    if return_data:
        return dslopeslist, vslopeslist
    return 0



#%% K^{-1} correlation lengths

def kinvplot_slopes(kilist, alist, pathlength, ymin=0, pt=14, savename='', \
                     plotslopes=False, startslope=20, end=100, verbose=False, \
                         showlegend=True,matchcolor=True, ymax=1,legendloc='best',\
                             plot=True, colorlist=cycle):
    '''
    Plots K^{-1} correlator and returns best fit inverse correlation length
    (= -slope after taking a logarithm)

    Parameters
    ----------
    kilist : np.array. 
        The different rows correspond to different values of the parameter a
        in 'alist'.
        Row i should be the K^{-1} values along the path, for parameter 
        value alist[i].
        
    alist : list of a values correponding to 'vlist'
    pathlength : int, how many points to plot
    
    [optional parameters] See 'visonplot_slopes' description.
    '''
    num_a = len(alist)
    
    # get inverse correlation lengths
    slopelist = []
    interceptlist = []

    for i in range(num_a):
        regvals = sp.stats.linregress(range(startslope, end+1), \
                                np.log(np.abs(np.array(kilist[i,:][startslope-1:end+1-1]))))
        if verbose:
            print(regvals)
            
        slope, intercept = regvals[:2]
        slopelist.append(slope)
        interceptlist.append(intercept)
    
    if plot:
        markerlist = ["^","s","d", "x","o","v","*"]
        for i in range(num_a):
            plt.plot(range(2,pathlength+2), kilist[i,:pathlength],label=r'$\alpha=%.2f$'%alist[i],\
                     marker=markerlist[i%len(markerlist)], linestyle='',fillstyle='none',\
                         color=colorlist[i%len(colorlist)])
        
        if plotslopes: # plot best fit slope
            xvals = np.arange(startslope,pathlength+2)
            slopecolor = colorlist if matchcolor else ['k']
            for i in range(num_a):
                plt.plot(xvals, np.exp(slopelist[i]*xvals+interceptlist[i]), \
                         color=slopecolor[i%len(slopecolor)])
        
        plt.yscale('log')
        if showlegend:
            plt.legend(fontsize=pt-2,labelspacing=0.2, handletextpad=0.1, \
                       borderpad=.2, handlelength=1, loc=legendloc)
        plt.xlabel(r'Vertex $v_\ell$ in zigzag path',fontsize=pt)
        plt.ylabel(r'$|K^{-1}(v_0,v_\ell)|$',\
                   fontsize=pt)
        plt.xticks(fontsize=pt-2)
        plt.yticks(fontsize=pt-2)
        
        if ymin > 0:
            plt.ylim(bottom=ymin)
        if ymax < 1:
            plt.ylim(top=ymax)
        
        if savename != '':
            plt.savefig(savename, bbox_inches='tight')
        
        plt.show()
    return slopelist


def kinv_corr_lengths(n, alist, startslope=30, end=100, pt=18,\
                 savename='',save=False,filepath='./',\
                    plot=True, return_data=False):
    '''
    Plots inverse correlation lengths for K^{-1} as a function of parameter 'a' in 'alist'
    Plots vertical path and horizontal path (colshift=0)
    
    Loads saved .npy dimer and vison correlator data.

    Parameters
    ----------
    n : int. system size.
    alist : list of a values correponding to 'vlist'
    startslope : edge/face distance to start at for the slope regression. The default is 20.
    end : edge/face distance to end the slope regression. The default is 100.
    pt : int. font size.
    savename : name to save figure to. If non-empty, will save even if 'save' is False.
    save : If True, save figure to file. If 'savename' is empty, will use a default name.
    filepath : path to dimer and vison .npy files.
    vert : If True, will try to load files for vertical path. 
           If False, will try to load files for horizontal path
    colshift : lattice path shift amount, for files to load.
    plot : If False, will not plot intermediate dimer-dimer and vison correlator plots.
    return_data : If True, will return the best-fit slopes for dimers and visons
    '''
    plt.rcParams["figure.figsize"] = (6.4,4.2)
    
    alist = np.sort(alist)
    kilist = load_ki(n,alist=alist,filepath=filepath,vert=True, colshift=0)
    kilist_h = load_ki(n,alist=alist,filepath=filepath,vert=False, colshift=0)
    
    kislopes = kinvplot_slopes(kilist, alist, end, startslope=startslope,end=end,\
                               ymin=0,plotslopes=True,showlegend=False,\
                                   plot=plot, matchcolor=False)
    kislopes_h = kinvplot_slopes(kilist_h, alist, end, startslope=startslope,end=end,\
                               ymin=0,plotslopes=True,showlegend=False,\
                                   plot=plot, matchcolor=False)
    plt.show()
    
    fig, ax = plt.subplots()
    cmap = mpl.colormaps['plasma']

    ax.plot(alist, -np.array(kislopes), color='blue', marker='o',\
            fillstyle='none',label='Vertical path')
    ax.plot(alist, -np.array(kislopes_h), color=cmap(.3), marker='s',\
            fillstyle='none',label='Horizontal path')    

    ax.set_ylabel('Inverse correlation length', fontsize=pt)
    ax.set_xlabel(r'$\alpha$', fontsize=pt)
    ax.tick_params(labelsize=pt-2)
    ax.tick_params(axis='y')
    
    ax.axvline(x=3,color='k',linestyle='dashed')
    ax.axhline(y=0,color='k',linestyle='dashed')
    
    xvals = np.linspace(2.25,3.75,50)
    ax.plot(xvals, np.abs(xvals-3)/4., color='k', label=r'$|\alpha-3|/4$')
    
    
    ax.legend(fontsize=pt-2,labelspacing=0.2, handletextpad=0.1, borderpad=.2, handlelength=1)
    
    if save==True or savename !='':
        if savename == '':
            savename = 'invcorrelationlength_n%i_%i-%i.pdf'%(n,startslope,end)
        plt.savefig(savename, bbox_inches='tight')
    
    plt.show()
    
    if return_data:
        return kislopes
    return 0

#%% finite size scaling

def save_visons_fs(n, alist= [0.5,1,2.5,3,3.5,4,5], pathlengthmin=50, \
                   vert=True, colshift=0, \
            filepath='./', verbose=True):
    '''
    Calculate and save vison and dimer correlators for all a values in 'alist',
    along path specified by 'triangle_string(n,n//2,vert,colshift)'
    
    INPUT
        n: system size
        alist: list of a values to run
        vert [optional]: If True, uses a vertical path. If False, uses horizontal path.
        colshift [optional]: Amount to shift the path by
        filepath [optional]: where to save npy files (relative file path)
        verbose [optional]: If True, prints progress when it finishes an a value.
        
    pathlength is in lattice coordinates
    '''
    if filepath[-1] != '/':
        filepath += '/'
    
    pathlength = max(pathlengthmin, n//2)
        
    vstring = 'v' if vert else 'h'
    for a in alist:
        v, d = vd_triangular(n,square_bool,pathlength,[a,1,1,1,1,1],verbose=False,\
                             plot=False,vert=vert,colshift=colshift)
        np.save(filepath + 'tri_visons_n%i_a%.2f_%s_shift%i.npy'%(n,a,vstring,colshift), v)
        np.save(filepath + 'tri_dimers_n%i_a%.2f_%s_shift%i.npy'%(n,a,vstring,colshift), d)
        
        if verbose:
            print('n=%i, saved visons and dimers for a=%.2f'%(n,a))
    
    return 0

def load_visons_fs(nvals, alist, pathloc, filepath='./', vert=True):
    '''
    Returns array of vison correlator values for values of n in nvals x values
    of a in alist
    Each row corresponds to one value of n
    
    INPUT:
        nvals: list of values of n to load
        alist: list of values of a to load (floats)
        pathloc: edge/face distance at which to get the vison correlator value
                set to a string (like 'half') to auto-set at edge distance n/2
                (halfway between center and edge)
        filepath: where to load data from
    '''
    if filepath[-1] != '/':
        filepath += '/'
    vh = 'v' if vert else 'h'
    
    vals = np.empty((len(nvals),len(alist)))
    for i, n in enumerate(nvals):
        for aindex, a in enumerate(alist):
            if type(pathloc) == str:
                pathlocval = n//2
            else:
                pathlocval = pathloc
            vals[i,aindex] = np.load(filepath+'tri_visons_n%i_a%.2f_%s_shift0.npy'%(n,a,vh))[pathlocval]
    return vals


def fs_scaling8(nvals, pathloc, filepath='./', beta=1., nu=1., pt=16,\
                savename='',textloc=(-50,.5), vert=True):
    '''
    Finite-size scaling plot with all 8 values of \alpha
    
    INPUT:
        nvals: list of values of n to plot
        pathloc: edge/face distance at which to get the vison correlator value
                 Set to a string (like 'half') to auto-set at edge distance n/2
                 (halfway between center and edge)
        filepath: where to load data from
        beta: critical exponent
        nu: critical exponent
        vert: True to load vertical path, otherwise load horizontal
    '''
    plt.rcParams["figure.figsize"] = (6.4,3.5)
    
    alist = [2.95,2.96,2.97,2.98,3.05,3.04,3.03,3.02]
    markerlist = ['o','s','^','*']
    mewidth = [1.5,1.5,1.5,1.5]
    cmap = matplotlib.cm.get_cmap('plasma')
    vison_vals = load_visons_fs(nvals, alist, pathloc, filepath, vert=vert)
    nvals = np.array(nvals)
    for aindex, a in enumerate(alist):
        vals = np.sqrt(vison_vals[:,aindex])
        mfc = cmap(aindex%4/4.) if a>3 else 'none'
        plt.plot((a-3)*(nvals-1)**(1./nu),vals*(nvals-1)**(beta/nu), linestyle='none',\
                 color=cmap(aindex%4/4.),marker=markerlist[aindex%4],\
                  #alpha=(10-aindex%4)/10., \
                markerfacecolor=mfc, markeredgewidth=mewidth[aindex%4],\
                label=r'$\alpha=%.2f$'%a)
    plt.xlabel(r'$(\alpha-3)N^{1/\nu}$', fontsize=pt)
    plt.ylabel(r'$V\cdot N^{\beta/\nu}$', fontsize=pt)
    plt.text(*textloc,r'$\beta=%.3f$, $\nu=%.2f$'%(beta,nu), fontsize=pt-2)
    plt.xticks(fontsize=pt-2)
    plt.yticks(fontsize=pt-2)
    plt.legend(fontsize=pt-2,labelspacing=0.2, handletextpad=0.2, borderpad=.2,\
        handlelength=1,ncol=2, columnspacing=.8)
    
    if savename !='':
        plt.savefig(savename, bbox_inches='tight')
    
    plt.show()

    return 0



#%% utility

def xy_dimensions(n, shape_bool):
    '''
    Return tuple (xwidth,ywidth) of the max lattice size in each direction.
    Currently, with only 'square_bool' and 'aztec_bool' defined, this is 
    a useless function which just returns (n,n).
    '''
    match shape_bool.__name__:
        case 'aztec_bool':
            xwidth = n
            ywidth = n

        case _:
            xwidth = n    
            ywidth = n
    return xwidth, ywidth

def plot_shape_bool(n, shape_bool, string=[],numbering=False, weights=False, \
               kmat='', axes=True, pt=16, edge_col='gray', kinv='',\
                   orientation=False,colormap='viridis', title='', savename=''):
    '''
    Draw the lattice and edges determined by 'shape_coord'.
    
    'string': a list of edges [[(v1,v2),(v3,v4)], [(w1,w2),(w3,w4)], ...]
    that will be highlighted in the drawing
    
    'numbering': If True, will number every vertex with its matrix coordinate
    
    'weights': If True, label each edge with its weight from the Kasteleyn 
               matrix. (If 'kmat' is not supplied, the weights will be the
                        uniform \pm 1 in the Kastleyn matrix)
        
    'kmat': Optional argument- If passed, this will be used for the Kasteleyn
            matrix. This is only useful with 'weights'=True, to plot non-uniform
            weights.
    
    Remember all plotting is backwards...i = vertical, j=horizontal,
    so for plt.plot, reverse the coords
    '''
    kast = kmat if type(kmat) != str else kasteleyn_tri21(n, shape_bool)
    xwidth, ywidth = xy_dimensions(n, shape_bool)
    
    shape_coord = shape_coord_from_bool(n, shape_bool)
    fig, ax = plt.subplots(figsize=(xwidth//n*8, ywidth//n*8))

    # line segments for the lattice
    line_segments = []
    edge_colors = []
    edgewidth = .7 # default value, changes for showing edge probabilities
                
    for i in range(ywidth):
        for j in range(xwidth):
            if is_in_shape(n,i,j,shape_coord): 
                coord = shape_coord(n,i,j)
                # check which neighbors are in the shape, and which are connected
                for neighbor in [(i+1,j),(i,j+1),(i+1,j+1)]:
                    if is_in_shape(n, neighbor[0], neighbor[1], shape_coord):
                        nematcoord = shape_coord(n,*neighbor)
                        kval = kast[coord, nematcoord]
                        if kval != 0:
                            edge_colors.append(edge_col)
                            line_segments.append([(j,i),(neighbor[1],neighbor[0])])
                            if weights:
                                plt.text((j+neighbor[1])/2,(i+neighbor[0])/2,np.abs(kval))
                                direction = (neighbor[1]-j, neighbor[0]-i)
                                if kval < 0: # swap direction
                                    direction = -direction[0], -direction[1]
                                smallshift = 0.2
                                plt.arrow((j+neighbor[1])/2,(i+neighbor[0])/2, \
                                          smallshift*direction[0], smallshift*direction[1],\
                                              shape='full', lw=0, length_includes_head=True, head_width=.1)
                            if orientation:
                                start, end = [(j,i), (neighbor[1],neighbor[0])] if kval > 0 \
                                    else [(neighbor[1],neighbor[0]), (j,i)]
                                vector = (end[0]-start[0])/2, (end[1]-start[1])/2
                                plt.arrow(*start, *vector, shape='full', lw=0, \
                                          length_includes_head=True, \
                                              head_width=0.12, color='k')
                                #print(start,end,coord,nematcoord,kast[coord,nematcoord])

                                    
                # if we want to number every vertex with its matrix coordinate
                if numbering == True:
                    plt.text(j, i, "%d" %coord, ha="center", fontsize=pt)
            # except:
            #     pass
    
    # DRAW
    ax.set_xlim(-1, xwidth)
    ax.set_ylim(-1, ywidth)
    ax.set_aspect("equal")
    
    # draw lattice edges
    #print(line_segments)
    line_collection = LineCollection(line_segments, colors=edge_colors, linewidths=edgewidth)
    ax.add_collection(line_collection)
    
    # draw the edges indicated in 'string'
    if len(string) > 0:
        for dimer in string:
            v1, v2 = dimer
            plt.plot([v1[1],v2[1]],[v1[0],v2[0]],'r-', linewidth=4)
    
    # make (0,0) the top left corner to agree with lattice numbering
    plt.xticks(fontsize=pt-2)
    plt.yticks(fontsize=pt-2)
    if axes == False:
        plt.axis('off')
        
    if len(title) > 0:
        plt.title(title)
    
    if savename != '':
        plt.savefig(savename, bbox_inches='tight')
    
    plt.show()
    return 0


def plot_points(points, log=True, title='',start=1, colors=False, xlog=False,\
                pt=15, label=[], savename='', takeabs=True):
    '''
    Plot the points in 'points'. This function is mainly useful if we need
    to set a logarithmic axes scale while using inline plotting.
    '''
    l = len(points)
    points = np.array(points)
    if takeabs == True:
        points = np.abs(points)
    
    if colors == False:
        plt.plot(range(start,l+start),points, '.', label=label)
    else: # colors = True => differentiate even from odd
        plt.plot(range(start, l+start,2),points[::2],'.')
        plt.plot(range(start+1,l+start,2),points[1::2],'.')
        
    if log==True:
        plt.yscale('log')
    if xlog == True:
        plt.xscale('log')
    if label != []:
        plt.legend()
    plt.xlabel('Position $\ell$ in string',fontsize=pt)
    plt.ylabel('',fontsize=pt)
    plt.title(title,fontsize=pt)
    plt.xticks(fontsize=pt-1)
    plt.yticks(fontsize=pt-1)
    
    if savename != '':
        plt.savefig(savename, bbox_inches='tight')
    
    plt.show()
    return 0

