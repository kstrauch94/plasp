
% Define unsubsumed mutexes

mutex(G,X) :- mutexGroup(G), contains(G,X,V). %, fluent(X,V).
mutex(G)   :- mutexGroup(G), #count{X : mutex(G,X)} > 1.

% Define initial state

%holds(X,V,0) :- initialState(X,V). %, fluent(X).
#external holds(X,V,T-1) : variable(X), contains(X,V), time(T).
init(holds(X,V,T-1)) :- initialState(X,V), time(T).

% generate (a SAT problem)
{     holds(X,V,T) } :- contains(X,V), time(T).
{ occurs(A,T) : action(A) } :- time(T).

:- {occurs(A,T): action(A)} > 1, time(T).

% effects
:- occurs(A,T), postcondition(A,E,X,V), not holds(X,V,T), time(T).

% preconditions
:- occurs(A,T), precondition(A,X,V),    not holds(X,V,T-1), time(T).%, time(T-1).

% inertia
:- holds(X,V,T), not holds(X,V,T-1), not occurs(A,T) : postcondition(A,_,X,V); time(T).%, time(T-1).

% existence and uniqueness of value
:- not 1 { holds(X,V,T) } 1, variable(X), time(T).

% mutexes
:- 2 #sum { 1,X,V: holds(X,V,T), contains(mutexGroup(G),X,V) }, mutex(G), time(T).


% Check goal conditions
#external query(T) : time(T).
goal(T) :- holds(X,V,T) : goal(X,V); time(T).
:- not goal(T), query(T).
#show occurs/2.
%#show holds/3.
#show goal/1.
#show query/1.
