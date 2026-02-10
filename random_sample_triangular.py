#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Random dimer coverings for the triangular lattice
Very inefficient, but can generate a few small random samples.  
"""

import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from kasteleyn_triangular import *

def gen_rectangle_bool(ratio):
    '''
    Return a shape_bool function lambda n,i,j: ...
    for a rectangle shape with height ratio*n (0<ratio\le1)
    '''
    def rshape(n,i,j):
        if i in range(int(ratio*n)) and j in range(n):
            return True
        return False
    return rshape

def rectangle_bool(n,i,j):
    '''
    rectangle shape, half the height of square to save space
    '''
    if i in range(n//2) and j in range(n):
        return True
    return False

def kasteleyn_tri21_lil(n, shape_bool, weightlist=[1]*6, dtype=float):
    '''    
    Create the Kasteleyn matrix for the triangular lattice with 2x1 periodic
    edge weights given by 'weightlist', where the edge order is
    [horizontal1, horizontal2, vertical1, vertical2, diagonal1, diagonal2]

    Output is a sparse lil_array.
    
    Same as construction from 'kasteleyn_triangular' but uses lil_array
    '''
        
    shape_coord = shape_coord_from_bool(n, shape_bool)
    # the number of vertices in the lattices
    size = num_vertices(n, shape_bool)
    width = n
    
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
                        # fewer edges (only relevant for Aztec diamond edges)
                        if direction != (1,1) or \
                            (direction == (1,1) and is_in_shape(n,i+1,j, shape_coord)\
                             and is_in_shape(n,i,j+1,shape_coord)): 
                            mat[coord, newcoord] = weight
                            mat[newcoord, coord] = -weight
                                                        
    return mat

def random_covering(n, shape_bool, weights=[1]*6):
    '''
    Generate a random dimer covering of the 2x1 triangular lattice.
    Output is a #vertices x 6 size array of integers
        - row is the vertex index (from the corresponding shape_coord)
        - columns represent the 6 edges adjacent to the vertex in order
            'step_list' = [(1,0),(1,1),(0,1),(-1,0),(-1,-1),(0,-1)]
        - value 1 = dimer present, 0 = no dimer
    
    Note: LU solver is recalculated for EVERY new edge AND converted to 
    csc every step. 
    This is VERY inefficient, but it is enough to generate a few small random
    samples. 
    
    Not recommended for larger sizes or repeated samples.
    '''
    kmat = kasteleyn_tri21_lil(n,shape_bool,weights) # lil_array is faster
    
    msize = num_vertices(n, shape_bool)
    shape_coord = shape_coord_from_bool(n, shape_bool)
    
    cover = np.full((msize,6), -1, dtype=np.int64)
    step_list = [(1,0),(1,1),(0,1),(-1,0),(-1,-1),(0,-1)]
    
    for y in range(n):
        for x in range(n):
            if shape_bool(n, y, x):
                mvertex = shape_coord(n, y, x)
                if 1 not in cover[mvertex,:]: # skip already paired vertices
                    lusolver = sps.linalg.splu(kmat)
                    cprob = np.zeros(6)
                    for index, direction in enumerate(step_list): # loop through the 6 edges for vertex (x,y)
                        neighbor = x+direction[0], y+direction[1]
                        if shape_bool(n, neighbor[1], neighbor[0]):
                            mneighbor = shape_coord(n, neighbor[1], neighbor[0])
                            
                            if cover[mvertex,index] != 0: # not a deleted edge
                                standard_basis_v = np.transpose(np.eye(1, msize, mvertex))
                                cprob[index] = np.abs(kmat[mvertex,mneighbor]*(lusolver.solve(standard_basis_v))[mneighbor,0])

                    if np.sum(cprob) < 0.99: # shouldn't ever be an issue
                        print(np.sum(cprob),cprob)
                        plot_cover(n, cover)
                    else:
                        cprob = cprob / np.sum(cprob)
                    # select dimer
                    new_edge_index = np.random.choice(range(6), p=cprob)
                    
                    newneighbor = x + step_list[new_edge_index][0], y+step_list[new_edge_index][1]
                    mnewneighbor = shape_coord(n,newneighbor[1], newneighbor[0])
                    
                    # update Kasteleyn matrix K
                    for mcoord in [mvertex,mnewneighbor]:
                        kmat[mcoord,:] = np.zeros(msize)
                        kmat[:,mcoord] = np.zeros(msize)
                    kmat[mvertex, mnewneighbor] = 1
                    kmat[mnewneighbor, mvertex] = -1
                    
                    # delete all non-chosen edges adjacent to (x,y)
                    for i2 in range(6):
                        if i2 != new_edge_index: # not the chosen edge
                            direction = step_list[i2]
                            notneighbor = x+direction[0], y+direction[1]
                            if shape_bool(n, notneighbor[1], notneighbor[0]):
                                mnotneighbor = shape_coord(n, notneighbor[1], notneighbor[0])
                                cover[mnotneighbor,(i2+3)%6] = 0
                    
                    # delete all non-chosen edges adjacent to neighbor
                    chosen_index = (new_edge_index+3)%6
                    for i3 in range(6):
                        if i3 != chosen_index:
                            direction = step_list[i3]
                            notneighbor = newneighbor[0]+direction[0], newneighbor[1]+direction[1]
                            if shape_bool(n, notneighbor[1], notneighbor[0]):
                                mnotneighbor = shape_coord(n, notneighbor[1], notneighbor[0])
                                cover[mnotneighbor,(i3+3)%6] = 0
                        
                    cover[mvertex,:] = np.full(6,0,dtype=np.int64)
                    cover[mnewneighbor,:] = np.full(6,0,dtype=np.int64)
                    cover[mvertex,new_edge_index] = 1
                    cover[mnewneighbor,(new_edge_index+3)%6] = 1
            
    return cover

def plot_cover(n, shape_bool, cover, axes=False):
    '''
    Draw the cover described by 'cover' (e.g. output from 'random_covering')
    '''
    step_list = [(1,0),(1,1),(0,1),(-1,0),(-1,-1),(0,-1)]

    line_segments = []
    shape_coord = shape_coord_from_bool(n, shape_bool)

    for x in range(n):
        for y in range(n):
            if shape_bool(n, y, x):
                mvertex = shape_coord(n, y, x)
                dindex = np.nonzero(cover[mvertex,:])[0][0]
                if cover[mvertex,dindex] > 0:
                    neighbor = x+step_list[dindex][0], y+step_list[dindex][1]
                    line_segments.append([(x,y),neighbor])
    
    ax = plt.gca()
    shift=1
    ax.set_xlim(-shift,n)
    ax.set_ylim(-shift,n)
    ax.set_aspect('equal', adjustable='box')
    
    if axes == False:
        plt.axis('off')
        
    # draw edges
    line_collection = LineCollection(line_segments, linewidths=1, colors='k')
    ax.add_collection(line_collection)
    plt.show()
    return 0


def plot_2covers(n, shape_bool, cover1, cover2, axes=False, savename='',\
                 loopcolor='k', decolor='darkgrey', dewidth=.5):
    '''
    Draw the two covers (overlayed) described by 'cover1' and 'cover2'
    (e.g. output from 'random_covering')
    
    OPTIONS:
        loopcolor - loop color
        decolor - double edge color
        dewith - width of double edges
    '''
    step_list = [(1,0),(1,1),(0,1),(-1,0),(-1,-1),(0,-1)]

    loop_segments = []
    double_edges = []
    shape_coord = shape_coord_from_bool(n, shape_bool)
                
    height = 0 # largest y-value to plot
    
    for x in range(n):
        for y in range(n):
            if shape_bool(n, y, x):
                height = max(height,y)
                mvertex = shape_coord(n, y, x)
                dindex1 = np.nonzero(cover1[mvertex,:])[0][0] # get index of paired vertex
                dindex2 = np.nonzero(cover2[mvertex,:])[0][0] # get index of paired vertex
                if cover1[mvertex,dindex1] > 0 and cover2[mvertex,dindex2] > 0:
                    neighbor1 = x+step_list[dindex1][0], y+step_list[dindex1][1]
                    if dindex1 == dindex2:
                        double_edges.append([(x,y),neighbor1])
                    else:
                        neighbor2 = x+step_list[dindex2][0], y+step_list[dindex2][1]
                        loop_segments.append([(x,y),neighbor1])
                        loop_segments.append([(x,y),neighbor2])
                else:
                    print('Error? x,y=%i,%i'%(x,y))
                    print(cover1[mvertex,:],cover2[mvertex,:])
    
    ax = plt.gca()
    shift=1
    ax.set_xlim(-shift,n)
    ax.set_ylim(-shift,height+1)
    ax.set_aspect('equal', adjustable='box')
    
    if axes == False:
        plt.axis('off')
        
    # draw edges
    loop_collection = LineCollection(loop_segments, linewidths=.8, colors=loopcolor)
    ax.add_collection(loop_collection)
    
    de_collection = LineCollection(double_edges, linewidths=dewidth, colors=decolor)
    ax.add_collection(de_collection)
    
    if savename != '':
        plt.savefig(savename, bbox_inches='tight')
    
    plt.show()
    return 0
