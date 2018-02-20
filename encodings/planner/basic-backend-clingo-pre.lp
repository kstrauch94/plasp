#include "../strips/preprocess.lp".
% Constant '_parallel' to enable parallel actions
% - value '0': parallel actions *without* checking for mutual disablement
% - value '1': "forall" parallel actions that can be arranged in any sequence
% - value '2': "exists" parallel actions that can be arranged in some sequence
% - value '3': "exists" parallel actions omitting achievement of preconditions
% - value '4': "exists" parallel actions like '3' implemented by acyclicity
% - otherwise: sequential actions

#const _parallel = 5.

% Constant '_shallow' to cheaply approximate mutually disabling parallel actions
% - value '0': detect incompatible pre/postconditions to reduce parallel actions
% - otherwise: over-approximate parallel actions (in testing mutual disablement)

#const _shallow = 0.

% Constant '_inertia' to choose flat or tiered format for transition constraints
% - value '0': use flat rules (integrity constraints) for transition constraints
% - otherwise: use rules that provide auxiliary atoms for transition constraints

#const _inertia = 0.

% Constant 'planner_on' to enable encoding support for ASP-based planner
% - value '0': no compatibility to ASP-based planner (no idle steps permitted)
% - otherwise: compatibility to ASP-based planner (idle steps permitted)

#const planner_on = 0.

% BASE PROGRAM

% Define auxiliary predicates for actions w.r.t. parallel mode

selfdefeat(A,X) :- active(A), precondition(A,X,V), _parallel = 1,
                   has_condition(A,X,1), not postcondition(A,X,V).

diverge(A1,A2,X) :- active(A1), active(A2), postcondition(A1,X,V),
                    has_condition(A2,X,1), not postcondition(A2,X,V),
                    0 <= _parallel, _parallel < 5, _shallow = 0.
diverge(A1,A2)   :- diverge(A1,A2,X).

exclude(A1,A2) :- active(A1), active(A2), precondition(A1,X,V),
                  has_condition(A2,X,0), not precondition(A2,X,V),
                  0 <= _parallel, _parallel < 5, _parallel != 2, _shallow = 0.

disable(A1,A2) :- active(A1), active(A2), A1 != A2, postcondition(A1,X,V),
                  has_condition(A2,X,0), not precondition(A2,X,V),
                  not diverge(A1,A2), not exclude(A1,A2),
                  1 < _parallel, _parallel < 5.

disable :- _parallel = 4, #sum{1,A1 : disable(A1,A2); -1,A2 : disable(A1,A2)} > 0.

scope(X,V) :- active(A), precondition(A,X,V), _parallel = 2.

% Define relevant fluents w.r.t. parallel mode

fluent(X,V) :- produce(X,V).
fluent(X,V) :- persist(X,V).
fluent(X,V) :- initialState(X,V), fluent(X).
fluent(X,V) :- active(A), postcondition(A,X,V), fluent(X).
fluent(X)   :- fluent(X,V).
fluent(X)   :- diverge(A1,A2,X), not exclude(A1,A2).
fluent(X)   :- active(A), has_condition(A,X,1), 0 <= _parallel, _parallel < 5, _shallow != 0.

% Define unsubsumed mutexes

mutex(G,X) :- mutexGroup(G), contains(G,X,V), fluent(X,V).
mutex(G)   :- mutexGroup(G), #count{X : mutex(G,X)} > 1.

% Define initial state

init(holds(X,V)) :- initialState(X,V), fluent(X).

% NEW: GENERATE STATE
#external holds'(X,V) : contains(X,V).

% STEP PROGRAM

%#program step(t).

% Generate successor state

1 {holds(X,V) : fluent(X,V)} 1 :- fluent(X).

:- mutex(G), #count{X,V : holds(X,V), contains(G,X,V)} > 1.

change(X) :- holds'(X,V), not holds(X,V), _inertia != 0.

% Generate actions

{occurs(A)} :- active(A).

:- planner_on = 0, not occurs(A) : active(A).

:- occurs(A), postcondition(A,X,V), fluent(X), not holds(X,V).

effect(X) :- occurs(A), postcondition(A,X,V), fluent(X), not precondition(A,X,V),
               _inertia != 0.

:- change(X), not effect(X).

:- _inertia = 0, holds(X,V), not holds'(X,V),
   not occurs(A) : active(A), postcondition(A,X,V), not precondition(A,X,V).

% Checks w.r.t. parallel mode

:- _parallel != 0, _parallel != 1, _parallel != 2, _parallel != 3, _parallel != 4, #count{A : occurs(A)} > 1.

:- _parallel != 2, occurs(A), precondition(A,X,V), not holds'(X,V).

:- _parallel = 1, occurs(A), precondition(A,X,V), not has_condition(A,X,1), not holds(X,V).

single(X) :- occurs(A), selfdefeat(A,X).

:- single(X), #count{A : occurs(A), postcondition(A,X,V), not precondition(A,X,V)} > 1.

proceed(X,V) :- holds'(X,V), scope(X,V).
proceed(X,V) :- perform(A), scope(X,V), postcondition(A,X,V), not precondition(A,X,V),
                  occurs(A).

perform(A) :- active(A), 1 < _parallel, _parallel < 4,
                proceed(X,V) : precondition(A,X,V), _parallel = 2;
                alright(A1) : disable(A,A1).

alright(A) :- active(A), 1 < _parallel, _parallel < 4, not occurs(A).
alright(A) :- perform(A).

:- 1 < _parallel, _parallel < 4, active(A), not alright(A).

#edge((A1),(A2)) : disable(A1,A2), occurs(A1), _parallel = 4, not disable.
#edge((A1),(A2)) : disable(A1,A2), occurs(A2), disable.

% CHECK PROGRAM

%#program check(t).

% Check goal conditions

% NEW: LAST ATOM
#external query.

:- query, goal(X,V), not holds(X,V).

% DISPLAY PART

show(occurs, 1).
