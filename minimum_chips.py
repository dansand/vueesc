
# coding: utf-8

# In[55]:

import underworld as uw
import math
from underworld import function as fn
import glucifer.pylab as plt
#import matplotlib.pyplot as pyplot
import time
import numpy as np
import os
import sys

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()


# In[56]:

ETA_T = 1e5
newvisc= math.exp(math.log(ETA_T)*0.53)


# In[ ]:




# In[57]:

###########
#Constants
###########
RA  = 1e2*newvisc       # Rayleigh number
TS  = 0          # surface temperature
TB  = 1          # bottom boundary temperature (melting point)
ETA_T = 1e5
ETA_Y = 10
ETA0 = 1e-3*newvisc
YSTRESS = 1.*newvisc
D = 2890.
MAXY = 1.0


# In[58]:

#Watch the type assignemnt on sys.argv[1]

DEFAULT = 64
    
if len(sys.argv) == 1:
    RES = DEFAULT
elif sys.argv[1] == '-f':
    RES = DEFAULT
else:
    RES = int(sys.argv[1])


# In[59]:

dim = 2          # number of spatial dimensions

Xres, Yres = RES, RES
dim = 2          # number of spatial dimensions


# In[60]:

yelsize = MAXY/Yres
yelsize*D


# Select which case of viscosity from Tosi et al (2015) to use. Adjust the yield stress to be =1 for cases 1-4, or between 3.0 and 5.0 (in increments of 0.1) in case 5.

# Set output file and directory for results

# In[61]:

if uw.rank()==0:
    print("############################" + "\n" + "Front matter Done" + "\n" + "############################")


# Create mesh objects. These store the indices and spatial coordiates of the grid points on the mesh.

# In[62]:

elementMesh = uw.mesh.FeMesh_Cartesian( elementType=("Q1/dQ0"), 
                                         elementRes=(Xres, Yres), 
                                           minCoord=(0.,0.), 
                                           maxCoord=(1.,MAXY))
linearMesh   = elementMesh
constantMesh = elementMesh.subMesh 


# In[63]:

if uw.rank()==0:
    print("############################" + "\n" + "Mesh initialised" + "\n" + "############################")


# In[64]:

velocityField    = uw.fevariable.FeVariable( feMesh=linearMesh,   nodeDofCount=dim )
pressureField    = uw.fevariable.FeVariable( feMesh=constantMesh, nodeDofCount=1 )
temperatureField = uw.fevariable.FeVariable( feMesh=linearMesh,   nodeDofCount=1 )


# In[65]:

if uw.rank()==0:
    print("############################" + "\n" + "FeVariable initialised" + "\n" + "############################")


# #ICs and BCs

# In[66]:

# Initialise data.. Note that we are also setting boundary conditions here
velocityField.data[:] = [0.,0.]
pressureField.data[:] = 0.
temperatureField.data[:] = 0.

# Setup temperature initial condition via numpy arrays
A = 0.01
#Note that width = height = 1
tempNump = temperatureField.data
for index, coord in enumerate(linearMesh.data):
    pertCoeff = (1- coord[1]) + A*math.cos( math.pi * coord[0] ) * math.sin( math.pi * coord[1] )
    tempNump[index] = pertCoeff;
    if coord[1] > 1:
        tempNump[index] = 0.


    


# In[67]:

# Get the actual sets 
#
#  HJJJJJJH
#  I      I
#  I      I
#  I      I
#  HJJJJJJH
#  
#  Note that H = I & J 

# Note that we use operator overloading to combine sets
IWalls = linearMesh.specialSets["MinI_VertexSet"] + linearMesh.specialSets["MaxI_VertexSet"]
JWalls = linearMesh.specialSets["MinJ_VertexSet"] + linearMesh.specialSets["MaxJ_VertexSet"]
TWalls = linearMesh.specialSets["MaxJ_VertexSet"]
BWalls = linearMesh.specialSets["MinJ_VertexSet"]


# In[ ]:




# In[68]:

# Now setup the dirichlet boundary condition
# Note that through this object, we are flagging to the system 
# that these nodes are to be considered as boundary conditions. 
# Also note that we provide a tuple of sets.. One for the Vx, one for Vy.
freeslipBC = uw.conditions.DirichletCondition(     variable=velocityField, 
                                              nodeIndexSets=(IWalls,JWalls) )

# also set dirichlet for temp field
tempBC = uw.conditions.DirichletCondition(     variable=temperatureField, 
                                              nodeIndexSets=(JWalls,) )


# In[69]:

# Set temp boundaries 
# on the boundaries
for index in linearMesh.specialSets["MinJ_VertexSet"]:
    temperatureField.data[index] = TB
for index in linearMesh.specialSets["MaxJ_VertexSet"]:
    temperatureField.data[index] = TS


# In[70]:

if uw.rank()==0:
    print("############################" + "\n" + "ICs and Bcs Done" + "\n" + "############################")


# #Particles

# In[71]:

# We create swarms of particles which can advect, and which may determine 'materials'
gSwarm = uw.swarm.Swarm( feMesh=elementMesh)

# Now we add a data variable which will store an index to determine material
materialVariable = gSwarm.add_variable( dataType="char", count=1 )


# Layouts are used to populate the swarm across the whole domain
# Create the layout object
layout = uw.swarm.layouts.GlobalSpaceFillerLayout( swarm=gSwarm, particlesPerCell=20)
# Now use it to populate.
gSwarm.populate_using_layout( layout=layout )




mantleIndex = 0
lithosphereIndex = 1
crustIndex = 2
airIndex = 3


# Set the material to heavy everywhere via the numpy array
materialVariable.data[:] = mantleIndex


# In[72]:

if uw.rank()==0:
    print("############################" + "\n" + "Particles set up" + "\n" + "############################")


# #Material Graphs

# In[73]:

##############
#Important: This is a quick fix for a bug that arises in parallel runs
##############
material_list = [0,1,2,3]


# In[74]:

#All depth conditions are given as (km/D) where D is the length scale, 
#note that 'model depths' are used, e.g. 1-z, where z is the vertical Underworld coordinate
#All temp conditions are in dimensionless temp. [0. - 1.]

#A few paramters defining lengths scales

Crust = 27.
CrustM = Crust/D

#Set initial air and crust materials (allow the graph to take care of lithsophere)

#########
#This initial material setup will be model dependent
#########
for particleID in range(gSwarm.particleCoordinates.data.shape[0]):
    if (1. - gSwarm.particleCoordinates.data[particleID][1]) < 0:
             materialVariable.data[particleID] = airIndex
    elif (1. - gSwarm.particleCoordinates.data[particleID][1]) < CrustM:
             materialVariable.data[particleID] = crustIndex



#######Setup some variables which help define condtions
#rock-air topography limits
dz = 50./D

avgtemp = 0.5


# In[75]:

import networkx as nx

#######Graph object
DG = nx.DiGraph(field="Depth")

#######Nodes
#Note that the order of materials, deepest to shallowest is important 
DG.add_node(0, mat='mantle')
DG.add_node(1, mat='lithosphere')
DG.add_node(2, mat='crust')
DG.add_node(3, mat='air')


labels=dict((n,d['mat']) for n,d in DG.nodes(data=True))
pos=nx.spring_layout(DG) 


#######Edges
#anything to air
DG.add_edges_from([(0,3),(1,3), (2,3)])
DG[0][3]['depthcondition'] = -1*dz
DG[1][3]['depthcondition'] = -1*dz
DG[2][3]['depthcondition'] = -1*dz


#Anything to mantle
DG.add_edges_from([(2,0), (3,0), (1,0)])
DG[3][0]['depthcondition'] = dz
DG[2][0]['depthcondition'] = (300./D)
DG[1][0]['depthcondition'] = (660./D) #This means we're going to kill lithosphere at the 660.


#Anything to lithsphere
DG.add_edges_from([(0,1),(3,1)])
DG[0][1]['depthcondition'] = 200./D
DG[0][1]['avgtempcondition'] = 0.75*avgtemp #definition of thermal lithosphere


#Anything to crust
DG.add_edges_from([(0,2), (1,2)])
DG[0][2]['depthcondition'] = CrustM
DG[1][2]['depthcondition'] = CrustM


# In[76]:

DG.nodes()


# In[77]:

remove_nodes = []
for node in DG.nodes():
    if not node in material_list:
        remove_nodes.append(node)
        
for rmnode in remove_nodes:
    DG.remove_node(rmnode)


# In[78]:

DG.nodes()


# In[79]:

#A Dictionary to map strings in the graph (e.g. 'depthcondition') to particle data arrays

particledepths = 1. - gSwarm.particleCoordinates.data[:,1]
particletemps = temperatureField.evaluate(gSwarm)[:,0]

conditionmap = {}

conditionmap['depthcondition'] = {}
conditionmap['depthcondition']['data'] = particledepths
conditionmap['avgtempcondition'] = {}
conditionmap['avgtempcondition']['data'] = particletemps


# In[80]:

def update_swarm(graph, particleIndex):
    """
    This function takes the materials graph (networkx.DiGraph), and a particle index,
    then determines if a material update is required 
    and if so, returns the new materialindex
    Args:
        graph (networkx.DiGraph): Directed multigraph representing the transformation of material types
        particleIndex (int): the particle index as corressponding to the index in the swarm data arrays
    Returns:
        if update is required the function returns the the new material variable (int) 
        else returns None
    Raises:
        TypeError: not implemented
        ValueError: not implemented
    """
    ##Egde gives links to other materials, we then query the conditions to see if we should change materials
    matId = materialVariable.data[particleIndex][0]
    innerchange = False
    outerchange = False
    for edge in graph[matId]:
        if outerchange:
            break
        for cond in graph[matId][edge].keys():
            outerchange = False
            if innerchange: #found a complete transition, break inner loop
                break
            currentparticlevalue = conditionmap[cond]['data'][particleIndex]
            crossover = graph[matId][edge][cond]
            if ((matId > edge) and (currentparticlevalue > crossover)):
                innerchange = False # continue on, 
                if graph[matId][edge].keys()[-1] == cond:
                    outerchange = True
                    innerchange = edge
                    break
            elif ((matId < edge) and (currentparticlevalue < crossover)):
                innerchange = False
                if graph[matId][edge].keys()[-1] == cond:
                    outerchange = True
                    innerchange = edge
                    break
            else:
                #condition not met, break outer loop, go to next edge, outerchange should still be False
                break
    if type(innerchange) == int:
        return innerchange


# for particleID in range(gSwarm.particleCoordinates.data.shape[0]):
#                 check = update_swarm(DG, particleID)
#                 #print check
#                 if check > -1:
#                     #number_updated += 1
#                     materialVariable.data[particleID] = check

# In[81]:

#Cleanse the swarm of its sins
#For some Material Graphs, the graph may have to be treaversed more than once

check = -1
number_updated = 1

while number_updated != 0:
    number_updated = 0
    for particleID in range(gSwarm.particleCoordinates.data.shape[0]):
                check = update_swarm(DG, particleID)
                if check > -1:
                    number_updated += 1
                    materialVariable.data[particleID] = check


# In[82]:

print("before comm.Barrier()")


# In[83]:

comm.Barrier()


# In[84]:

print("After comm.Barrier()")


# In[85]:

if uw.rank()==0:
    print("############################" + "\n" + "Material graphs done" + "\n" + "############################")


# #Material properties
# 

# In[ ]:




# In[86]:

#Make variables required for plasticity

secinvCopy = fn.tensor.second_invariant( 
                    fn.tensor.symmetric( 
                        velocityField.gradientFn ))

coordinate = fn.input()

depth = 1. - coordinate[1]


# In[87]:

depthField = uw.fevariable.FeVariable( feMesh=linearMesh,   nodeDofCount=1 )

depthField.data[:] = depth.evaluate(linearMesh)
depthField.data[np.where(depthField.data[:] < 0.)[0]] = 0.
#depthdata = depth.evaluate(linearMesh)


# In[ ]:




# In[88]:


viscosityl2 = newvisc*fn.math.exp((math.log(ETA_T)*-1*temperatureField) + (depthField*math.log(ETA_Y)))

viscosityFn1 = viscosityl2 #This one always gets passed to the first velcotity solve

#Von Mises effective viscosity
viscosityp = ETA0 + YSTRESS/(secinvCopy/math.sqrt(0.5)) #extra factor to account for underworld second invariant form


viscosityFn2 = 2./(1./viscosityl2 + 1./viscosityp)


# In[89]:

#Compositional Rayligh number of rock-water 

g = 9.81
rho = 3300
a = 1.25*10**-5
kappa = 10**-6
dT = 2500
eta0 = rho*g*a*dT*((D*1e3)**3)/(RA*kappa)
#Composisitional Rayleigh number
Rc = (3300*g*(D*1000)**3)/(eta0*kappa)


# In[90]:

CompRAfact = Rc/RA

airviscosity = 0.001*viscosityl2.evaluate(linearMesh).min()
airdensity = RA*CompRAfact


# In[91]:

##This block sets up rheolgoy for models with crust rheology;

#Von Mises effective viscosity
crustviscosityp = ETA0 + ((0.1*YSTRESS)/(secinvCopy/math.sqrt(0.5))) #extra factor to account for underworld second invariant form
crustviscosityFn2 = 2./(1./viscosityl2 + 1./crustviscosityp)


# Set up simulation parameters and functions
# ====
# 
# Here the functions for density, viscosity etc. are set. These functions and/or values are preserved for the entire simulation time. 

# In[92]:

# Here we set a viscosity value of '1.' for both materials
viscosityMapFn = fn.branching.map( keyFunc = materialVariable,
                         mappingDict = {airIndex:airviscosity, lithosphereIndex:viscosityFn2, crustIndex:viscosityFn2,mantleIndex:viscosityFn2} )

densityMapFn = fn.branching.map( keyFunc = materialVariable,
                         mappingDict = {airIndex:airdensity, lithosphereIndex:RA*temperatureField, crustIndex:RA*temperatureField, mantleIndex:RA*temperatureField} )

# Define our gravity using a python tuple (this will be automatically converted to a function)
gravity = ( 0.0, 1.0 )

buoyancyFn = gravity*densityMapFn


# In[93]:

if uw.rank()==0:
    print("############################" + "\n" + "Material properties done" + "\n" + "############################")


# Build the Stokes system, solvers, advection-diffusion
# ------
# 
# Setup linear Stokes system to get the initial velocity.

# In[94]:

#We first set up a l
stokesPIC = uw.systems.Stokes(velocityField=velocityField, 
                              pressureField=pressureField,
                              conditions=[freeslipBC,],
#                              viscosityFn=viscosityFn1, 
                              viscosityFn=fn.exception.SafeMaths(viscosityFn1), 
                              bodyForceFn=buoyancyFn)


# We do one solve with linear viscosity to get the initial strain rate invariant. This solve step also calculates a 'guess' of the the velocity field based on the linear system, which is used later in the non-linear solver.

# In[95]:

stokesPIC.solve()


# In[96]:

# Setup the Stokes system again, now with linear or nonlinear visocity viscosity.
stokesPIC2 = uw.systems.Stokes(velocityField=velocityField, 
                              pressureField=pressureField,
                              conditions=[freeslipBC,],
                              viscosityFn=fn.exception.SafeMaths(viscosityMapFn), 
                              bodyForceFn=buoyancyFn )


# In[97]:

solver = uw.systems.Solver(stokesPIC2) # altered from PIC2

solver.options.main.Q22_pc_type='uwscale'  # also try 'gtkg', 'gkgdiag' and 'uwscale'
solver.options.main.penalty = 1.0
solver.options.A11.ksp_rtol=1e-6
solver.options.scr.ksp_rtol=1e-5
solver.options.scr.use_previous_guess = True
solver.options.scr.ksp_set_min_it_converge = 1
solver.options.scr.ksp_set_max_it = 100
solver.options.mg.levels = 5
solver.options.mg.mg_levels_ksp_type = 'chebyshev'
solver.options.mg_accel.mg_accelerating_smoothing = True
solver.options.mg_accel.mg_accelerating_smoothing_view = False
solver.options.mg_accel.mg_smooths_to_start = 1


# Solve for initial pressure and velocity using a quick non-linear Picard iteration
# 

# In[98]:

solver.solve(nonLinearIterate=True)


# In[99]:

if uw.rank()==0:
    print("############################" + "\n" + "Solvers set up" + "\n" + "############################")


# Create an advective-diffusive system
# =====
# 
# Setup the system in underworld by flagging the temperature and velocity field variables.

# In[100]:

#Create advdiff system
advDiff = uw.systems.AdvectionDiffusion( temperatureField, velocityField, diffusivity=1., conditions=[tempBC,] )

advector = uw.systems.SwarmAdvector( swarm=gSwarm, velocityField=velocityField, order=1)


# In[101]:

if uw.rank()==0:
    print("############################" + "\n" + "Advection diffusion set up" + "\n" + "############################")


# Main simulation loop
# =======
# 
# The main time stepping loop begins here. Before this the time and timestep are initialised to zero and the output statistics arrays are set up. Also the frequency of outputting basic statistics to the screen is set in steps_output.
# 

# In[102]:

swarm_update = min(20, np.floor(10.*RES/64))


# In[103]:

realtime = 0.
step = 0
timevals = [0.]


# In[104]:

# initialise timer for computation
start = time.clock()
# setup summary output file (name above)
# Perform steps
while realtime < 0.15:
#while step < 4:
    #Enter non-linear loop
    solver.solve(nonLinearIterate=True)
    dt = advDiff.get_max_dt()
    if step == 0:
        dt = 0.
    advDiff.integrate(dt)
    # Advect swarm using this timestep size
    advector.integrate(dt)
    # Increment
    realtime += dt
    step += 1
    timevals.append(realtime)
    
    
        ################
    #Particle update
    ###############
    particledepths = 1. - gSwarm.particleCoordinates.data[:,1]
    particletemps = temperatureField.evaluate(gSwarm)[:,0]
    conditionmap['depthcondition']['data'] = particledepths
    conditionmap['avgtempcondition']['data'] = particletemps
    if step % swarm_update == 0:
        number_updated = 0
        for particleID in range(gSwarm.particleCoordinates.data.shape[0]):
            check = update_swarm(DG, particleID)
            if check > -1:
                number_updated += 1
                #if check == 0:
                #    print "from " + str(materialVariable.data[particleID]) + " to " + str(check)
                materialVariable.data[particleID] = check
            else:
                pass
        #Also update those integration swarms
        rockIntVar.data[:] = 0.
        notair = np.where(materialVariable.data != airIndex)
        rockIntVar.data[notair] = 1.
        airIntVar.data[:] = 0.
        notrock = np.where(materialVariable.data == airIndex)
        airIntVar.data[notrock] = 1.
        lithIntVar.data[:] = 0.
        islith = np.where((materialVariable.data == lithosphereIndex) | (materialVariable.data == crustIndex))
        lithIntVar.data[islith] = 1.
        #Also print some info at this step increment
        #print('steps = {0:6d}; time = {1:.3e}; v_rms = {2:.3f}; Nu0 = {3:.3f}; Nu1 = {3:.3f}'
        #  .format(step, realtime, Rms, float(Nu0glob), float(Nu1glob)))
    
        

if uw.rank() == 0:
    print("all done, time taken: ")
    print(time.clock()-start, realtime)


# In[107]:

#viscVariable = gSwarm.add_variable( dataType="float", count=1 )
#viscVariable.data[:] = viscosityMapFn.evaluate(gSwarm)
#figEta = plt.Figure()
#figEta.Points(gSwarm,viscVariable)
#figEta.VectorArrows(velocityField, linearMesh, lengthScale=0.02)
#figEta.Points(gSwarm,materialVariable, colours='brown white blue')
#figEta.Points(gSwarm,materialVariable)
#figEta.show()


# In[ ]:



