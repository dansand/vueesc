
# coding: utf-8

# In[1]:

# Blankenbach benchmark tests
# ======
#
# Case 2a: Temperature Dependent Convection
# ----
#

import underworld as uw
import math
from underworld import function as fn
import glucifer.pylab as plt
import numpy as np
import os
import time
import sys


# In[2]:

outputPath = "results"
outputFile = "Blankenbach"
if not os.path.exists(outputPath):
    os.makedirs(outputPath)

RES = 96

nproc = int(sys.argv[1])


dim = 2
Box_X = 1.0
Box_Y = 1.0
Box_Z = 1.0
DeltaTemp = 1.0
Temp_Min = 0.0
Temp_Max = DeltaTemp + Temp_Min
BoxSize = (Box_X, Box_Y)


# In[3]:

elementMesh = uw.mesh.FeMesh_Cartesian( elementType=('Q1/dQ0'),
                                         elementRes=(RES,RES),
                                           minCoord=(0.,0.),
                                           maxCoord=BoxSize)
velocityMesh    = elementMesh
temperatureMesh = elementMesh
pressureMesh    = elementMesh.subMesh

velocityField    = uw.fevariable.FeVariable( feMesh=velocityMesh,    nodeDofCount=dim )
temperatureField = uw.fevariable.FeVariable( feMesh=temperatureMesh, nodeDofCount=1 )
pressureField    = uw.fevariable.FeVariable( feMesh=pressureMesh,    nodeDofCount=1 )

velocityField.data[:] = [0.,0.]
pressureField.data[:] = 0.
temperatureField.data[:] = 0.


# In[4]:

pertStrength = 0.2
for index, coord in enumerate(temperatureMesh.data):
    pertCoeff = math.cos( math.pi/2.0 * coord[0]) * math.sin( math.pi * coord[1] )
    temperatureField.data[index] = Temp_Min + DeltaTemp*(1.0 - coord[1]) + pertStrength * pertCoeff
    temperatureField.data[index] = max(Temp_Min, min(Temp_Max, temperatureField.data[index]))

for index in temperatureMesh.specialSets["MinJ_VertexSet"]:
    temperatureField.data[index] = Temp_Max
for index in temperatureMesh.specialSets["MaxJ_VertexSet"]:
    temperatureField.data[index] = Temp_Min


# In[5]:

IWalls = elementMesh.specialSets["MinI_VertexSet"] + elementMesh.specialSets["MaxI_VertexSet"]
JWalls = elementMesh.specialSets["MinJ_VertexSet"] + elementMesh.specialSets["MaxJ_VertexSet"]
#KWalls = elementMesh.specialSets["MinK_VertexSet"] + elementMesh.specialSets["MaxK_VertexSet"]

freeslipBC = uw.conditions.DirichletCondition( variable=velocityField,
                                              nodeIndexSets=(IWalls,JWalls) )
tempBC = uw.conditions.DirichletCondition(     variable=temperatureField,
                                              nodeIndexSets=(JWalls,) )


# In[6]:

gSwarm = uw.swarm.Swarm( feMesh=elementMesh )
materialVariable = gSwarm.add_variable( dataType="int", count=1 )
gLayout = uw.swarm.layouts.GlobalSpaceFillerLayout( swarm=gSwarm, particlesPerCell=20 )
gSwarm.populate_using_layout( layout=gLayout )

materialIndex = 1
materialVariable.data[:] = materialIndex


# In[7]:

Ra = 1e7
eta0 = 1.0e3
dEta = 1.0e3

b = math.log(dEta)
T = temperatureField
viscosityFn = eta0 * fn.math.exp( -1.0 * b * T )

densityFn = Ra*temperatureField
gravity = ( 0.0, 1.0)
buoyancyFn = gravity*densityFn


# In[8]:

#stokesPIC = uw.systems.Stokes(velocityField=velocityField,
#                              pressureField=pressureField,
#                              swarm=gSwarm,
#                              conditions=[freeslipBC,],
#                              viscosityFn=viscosityFn,
#                              bodyForceFn=buoyancyFn )


stokesPIC = uw.systems.Stokes(velocityField=velocityField,
                              pressureField=pressureField,
                              swarm=gSwarm,
                              conditions=[freeslipBC,],
                              viscosityFn=viscosityFn,
                              bodyForceFn=buoyancyFn )

solver = uw.systems.Solver(stokesPIC)

solver.options.main.Q22_pc_type='uw'
solver.options.main.penalty = 0.0
solver.options.A11.ksp_rtol=1e-6
solver.options.scr.ksp_rtol=1e-5
solver.options.scr.use_previous_guess = True
solver.options.scr.ksp_set_min_it_converge = 1

solver.options.mg.levels = 5
solver.options.mg.mg_levels_ksp_type = 'chebyshev'
solver.options.mg_accel.mg_accelerating_smoothing = True
solver.options.mg_accel.mg_accelerating_smoothing_view = False
solver.options.mg_accel.mg_smooths_to_start = 1
solver.options.A11.ksp_rtol=1e-6
solver.options.mg.mg_levels_ksp_max_its = 3
solver.options.mg.mg_coarse_ksp_max_it = 3
solver.options.mg.pc_mg_smoothup = 5
solver.options.mg.pc_mg_smoothdown = 5

advDiff = uw.systems.AdvectionDiffusion( temperatureField, velocityField, diffusivity=1., conditions=[tempBC,] )

advector = uw.systems.SwarmAdvector( swarm=gSwarm, velocityField=velocityField, order=2 )

# Main simulation loop
# =======

realtime = 0.
step = 0
steps_end = 250
steps_output = min(1,steps_end/1)
steps_output = max(steps_output,1)
iloop = True


start = time.clock()
# Perform steps
while iloop == True:
    # Get solution for initial configuration
    stokesPIC.solve(nonLinearIterate=False)
    # Retrieve the maximum possible timestep for the AD system.
    dt = advDiff.get_max_dt()
    advector.integrate(dt)
    if step == 0:
        dt = 0.
    # Advect using this timestep size
    advDiff.integrate(dt)
    #if step%(steps_end/steps_output) == 0:
    #    temperatureField.save(outputPath+"/tempfield_out."+str(step))
    if step>=steps_end:
        iloop = False
    realtime += dt
    step += 1

if uw.rank() == 0:
    machine_time = (time.clock()-start)
    with open("results/blankenbach.txt", "ab") as text_file:
        text_file.write("\nMachineTime,%f,nproc,%i" % (machine_time,nproc))
