#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 17:20:49 2020

@author: joel
"""
import numpy as np
from numpy.random import random_sample
from operator import attrgetter
from time import time

class Particle():
    def __init__(self, pos, vel, b_pos, bounds):
        self.pos = pos.copy()
        self.vel = vel.copy()
        self.b_pos = b_pos.copy()
        self.fitness = np.inf
        self.bp_fitness = np.inf
        self.bounds = bounds
        self.eval = True

    def _checkPosition(self,b_handling='none'):
        """
        Check positions and handles bounds violations

        Parameters
        ----------
        i : Int
        Index of particle to check position.

        Returns
        -------
        list
        Position in a way that avoids bugs with Pandas.

        """
        self.eval = True
        n_pos = self.pos +self.vel
        if b_handling == 'none':
            self.pos = n_pos.copy()
            return
        elif b_handling == 'inf':
            for k, values in enumerate(self.bounds.values()):
                if n_pos[k] < values[0] or n_pos[k] > values[1]:
                    self.eval = False
                    self.pos += self.vel
                    self.fitness = np.inf
                    return
        elif b_handling == 'nearest':
            for k, values in enumerate(self.bounds.values()):
                if n_pos[k] < values[0]:
                    self.pos[k] = values[0]
                elif n_pos[k] > values[1]:
                    self.pos[k] = values[1]
                else:
                    self.pos[k] = n_pos[k]
        elif b_handling == 'pbest':
            for k, values in enumerate(self.bounds.values()):
                if n_pos[k] < values[0] or n_pos[k] > values[1]:
                    self.pos = self.b_pos.copy()
                    return
        elif b_handling == 'reflect':
            for k, values in enumerate(self.bounds.values()):
                if n_pos[k] < values[0]:
                    self.pos[k] = 2*values[0]-n_pos[k]
                elif n_pos[k] > values[1]:
                    self.pos[k] = 2*values[1]-n_pos[k]
                else:
                    self.pos[k] = n_pos[k]

        elif b_handling == 'random':
            for k, values in enumerate(self.bounds.values()):
                if n_pos[k] < values[0] or n_pos[k] > values[1]:
                    rand_pos = random_sample(len(n_pos))
                    for j, lims in enumerate(self.bounds.values()):
                        rand_pos[j] = (lims[1] - lims [0])*rand_pos[j] + lims[0]
                    self.pos = rand_pos.copy()
                    return






class PSOOptimizer():
    """
    PSO class.

    Parameters
    ----------
    func : Python Function
            Function to be optimized. Must return a scalar (fitness)

    bounds : Dict
            Bounds of every dimension. Must be a dict of type
            { 'name_of_dimension': [inf_lim, sup_lim] }

    w : Scalar, optional
            Inertia factor. The default is 0.9.

    s_fac : Scalar, optional
            Social factor of the particles. The default is 2.05.
    c_fac : Scalar, optional
            Cognitive factor of the particles. The default is 2.05.
    k     : Boolean, optional
            Factor with which velocities are multiplied. The default is True
    Returns
    -------
    None.

    """

    def __init__(self, func, bounds, w=0.9, s_fac=2.05, c_fac=2.05, k=True,**kwargs):
       self.func = func
       self.bounds = bounds
       self.w = w
       self.s_fac = s_fac
       self.c_fac = c_fac
       self.population = None
       self.g_pos = None
       self.fg_pos = np.inf
       self.dec_w = 0
       self.history = {}
       self.kwargs = kwargs
       self.time_elapsed = 0
       if k and (s_fac + c_fac > 4):
           phi = s_fac + c_fac
           self.k = 2/np.abs(2-phi-np.sqrt(phi**2 - 4*phi))
       elif k and (s_fac + c_fac < 4):
           print('If k == True then social factor + cognitive factor must be > 4')
           print('k has been set to False')
           self.k = 0
       else:
           self.k = 0



    def optimize(self, n_part, n_iter,b_handling='none'):
        """
        This function optimize the function given

        Parameters
        ----------
        n_part : int
            Number of particles to use during optimization
        n_iter : int
            Number of generations to use during optimization

        Returns
        -------
        None.

        """
        initial = time()
        if self.population:
            self._clearPopulation()
        self.history = {}
        self.g_pos = None
        self.fg_pos = np.inf
        self.dec_w = self.w/n_iter
        self.ni = 0


        self._createPopulation(n_part)
        #for p in self.population:
        #    print(p.pos)
        #    print(p.fitness)
        #print()
        for i in range(n_iter):
            self._movePopulation(n_part, self.k, b_handling)
        final = time()
        self.time_elapsed = final - initial


    def _createPopulation(self, n_part):
        """
        Creates population of n particles

        Parameters
        ----------
        n_part : int
            Number of particles to create the population with.

        Returns
        -------
        None

        """


        self.population = []

        dim = len(self.bounds.values())

        rand_mat = random_sample((n_part, dim))
        for j, lims in enumerate(self.bounds.values()):
            rand_mat[:,j] = (lims[1] - lims [0])*rand_mat[:,j] + lims[0]

        for i in range(n_part):
            self.population.append(Particle(rand_mat[i,:], random_sample(dim), rand_mat[i,:],self.bounds))


        idx_min = self._evaluatePopulation(n_part)
        if self.population[idx_min].fitness < self.fg_pos:
            self.g_pos = self.population[idx_min].pos
            self.fg_pos = self.population[idx_min].fitness
        self._getPopInfo()
        self.ni += 1
    def _evaluatePopulation(self, npart):
        """
        Evaluate the current population

        Returns
        -------
        Idx min of the fitness

        """
        for i in range(npart):
            if self.population[i].eval:
                fitness = self.func(self.population[i].pos,self.kwargs)
                self.population[i].fitness = fitness
            if self.population[i].fitness < self.population[i].bp_fitness:
                self.population[i].b_pos = self.population[i].pos.copy()
                self.population[i].bp_fitness = fitness

        return self.population.index(min(self.population, key=attrgetter('fitness')))


    def _movePopulation(self, npart, k, b_handling):
        """
        Moves every particle

        Returns
        -------
        None.

        """

        self._calc_new_velocities(npart, k)
        self._calc_new_positions(npart, b_handling)

        idx_min = self._evaluatePopulation(npart)
        if self.population[idx_min].fitness < self.fg_pos:
            self.g_pos = self.population[idx_min].pos
            self.fg_pos = self.population[idx_min].fitness
        self._getPopInfo()
        self.ni += 1



    def _calc_new_velocities(self, npart, k):
        """
        Function to calculate new velocities

        Parameters
        ----------
        npart : Int
            Number of particles.
        k : Float
            Factor with which velocites are multiplied.

        Returns
        -------
        None.

        """
        rn = random_sample(2*npart)

        if k:
            for i in range(npart):
                p_f = self.c_fac*rn[2*i]*(self.population[i].b_pos-self.population[i].pos)
                s_f = self.s_fac*rn[2*i+1]*(self.g_pos-self.population[i].pos)
                n_vel = k * (self.w*self.population[i].vel + s_f + p_f)
                self.population[i].vel = n_vel
        else:
            for i in range(npart):
                p_f = self.c_fac*rn[2*i]*(self.population[i].b_pos-self.population[i].pos)
                s_f = self.s_fac*rn[2*i+1]*(self.g_pos-self.population[i].pos)
                n_vel = self.w*self.population[i].vel + s_f + p_f
                self.population[i].vel = n_vel

        self.w -= self.dec_w

    def _calc_new_positions(self,npart,b_handling):
        """
        Function to calculate new positions

        Parameters
        ----------
        npart : Int
            Number of particles.

        Returns
        -------
        None.

        """
        for i in range(npart):
            self.population[i]._checkPosition(b_handling)




    def _clearPopulation(self):
        """
        Makes population == None

        Returns
        -------
        None.
        """

        self.population = None




    def _getPopInfo(self):
        """
        Function to get Population information. In the future will be used
        to create optimization logs.

        Returns
        -------
        None.

        """
        pos = []
        vel = []
        b_pos = []
        fitness = []
        bp_fitness = []
        for p in self.population:
            pos.append(p.pos)
            vel.append(p.vel)
            b_pos.append(p.b_pos)
            fitness.append(p.fitness)
            bp_fitness.append(p.bp_fitness)


        self.history[self.ni] = {'Pos':pos,'Vel':vel,'B_pos':b_pos,
                                'Fitness':fitness, 'BP_fitness':bp_fitness,
                                'GB_pos':self.g_pos,'GB_fitness':self.fg_pos}

if __name__ == "__main__":

    bounds = {'x': [-10,10],
              'y': [-10,10],
              'z': [-10,10],
              'a': [-10,10],
              'b': [-10,10],
              'c': [-10,10]}



    def suma(x):
        r = 0
        for c in x:
            r += c

        return r

    def sphere(x,*args):
        r = 0
        for c in x:
            r+= c**2

        return r

    def avg_fitness(f,bounds,b_handling, times):
        fitness_arr = np.empty(times)
        for i in range(times):
            pso = PSOOptimizer(f, bounds)
            pso.optimize(30,30,b_handling=b_handling)
            fitness_arr[i] = pso.fg_pos
            print('Iteración: ',i+1)
            print(pso.g_pos)
            print(pso.fg_pos)
            print()
        print('Se utilizo la tecnica: ', b_handling)
        print('El fitness promedio es: ', fitness_arr.mean())


    avg_fitness(sphere,bounds,'reflect', 1)
