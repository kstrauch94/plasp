#!/bin/bash

INSTANCE1=../../instances/PDDL/ipc-2002-depots-strips/problem-02.pddl

INSTANCE2=../../instances/PDDL/ipc-2004-satellite/problem-01.pddl
INSTANCE3=../../instances/PDDL/ipc-2004-satellite/problem-02.pddl

INSTANCE4=../../instances/PDDL/ipc-2000-blocks-track-1-typed/problem-09-02.pddl

INSTANCE5=../../instances/PDDL/ipc-1998-gripper/problem-01.pddl
INSTANCE6=../../instances/PDDL/ipc-1998-gripper/problem-02.pddl

INSTANCE7=../../instances/PDDL/ipc-2000-elevator-m10-strips/problem-04-01.pddl

INSTANCE8=../../instances/PDDL/ipc-2006-tpp-propositional/problem-04.pddl

INSTANCE9=../../instances/PDDL/ipc-2002-driver-log-strips/problem-03.pddl

INSTANCE10=../../../instances-easy/ipc-2002/domains/depots-strips-automatic/instances/instance-7.pddl

INSTANCE11=../../../instances-hard/ipc-2002/domains/depots-strips-automatic/instances/instance-13.pddl

INSTANCE12=../../../instances-easy/ipc-2002/domains/driverlog-strips-automatic/instances/instance-14.pddl



python -B runplanner.py $INSTANCE12 --translate --parallel=0
#python -B runplanner.py $INSTANCE12 --translate --parallel=0 --heuristic=Domain heuristic.lp

