#!/bin/usr/python

from __future__ import print_function
import clingo
import sys
from collections import namedtuple
from time import clock, time
from math import copysign

import subprocess

from tempfile import NamedTemporaryFile
import os

# DEFINES
STR_UNSAT = "error: input program is UNSAT"
GROUNDING_ERROR = "error: invalid grounding steps"
INIT_INCORRECT = "error: invalid init atom"
INIT = "init"
# mapping
PRIMED_EXTERNAL = -1
TRUE  = -2
FALSE = -3
# fitting
FITTING_TRUE     = 1
FITTING_FALSE    = 2
FITTING_CAUTIOUS = 3      

# log
log_level = 1
def log(*args):
    if log_level == 1:
        print(*args)

#
# Syntax Restriction:
#   - primed atoms are not allowed in the heads
#
# Semantics:
#   - the transition is defined by the stable models of the program
#   - the possible initial states are defined by the set of primed externals
#   - normal (not primed) externals work as usual
#
# Extensions (made by the controller object):
#   - init/1 defines the initial situation
#   - external last/0 is set to true only at the last step
#

class DLPGenerator:

    def __init__(self, files = [], adds = [], parts = [], options = []):
        # input
        self.files = files
        self.adds = adds
        self.parts = parts
        self.options = options
        # output
        self.offset = 0
        self.rules = []
        self.weight_rules = []
        self.primed_externals = {}
        self.normal_externals = {}
        self.output = []
        self.output_facts = []
        self.init = []
        # rest
        self.ctl = None
        self.next = {}
        self.mapping = []
        self.solve_for_output = True

    def run(self):
        # preliminaries
        ctl = self.ctl = clingo.Control(self.options)
        ctl.register_observer(self)
        for i in self.files:
            ctl.load(i)
        for i in self.adds:
            ctl.add(i[0], i[1], i[2])
        ctl.ground(self.parts)
        #print(self)
        # analyze
        self.set_externals()
        self.simplify()
        self.set_next()
        self.set_mapping()
        self.map_rules()
        self.map_weight_rules()
        self.handle_externals()
        self.set_output()
        self.set_init()
        # return
        return DynamicLogicProgramContainer(
            self.offset, self.rules, self.weight_rules,
            self.primed_externals, self.normal_externals,
            self.output, self.output_facts, self.init
        )

    def set_externals(self):
        # check symbolic atoms for externals and add them 
        # to either primed or normal externals list
        for x in self.ctl.symbolic_atoms:
            if x.is_external:
                self.ctl.assign_external(x.symbol, None)
                if len(x.symbol.name) and x.symbol.name[-1]=="'":
                    self.primed_externals[x.symbol] = x.literal
                else:
                    self.normal_externals[x.symbol] = x.literal

    def simplify(self):
        pass

    def get_next_literal(self, symbol, default):
        try:
            next_symbol = clingo.Function(symbol.name[:-1], symbol.arguments)
            return self.ctl.symbolic_atoms[next_symbol].literal
        except Exception:
            return default

    def set_next(self):
        self.next = {literal : self.get_next_literal(symbol, literal)
                     for symbol, literal in self.primed_externals.items() }

    def set_mapping(self):
        self.mapping = [0] * (self.offset + 1) # TODO: check if offset works
        for i in range(1, self.offset + 1):
            self.mapping[i] = self.next.get(i, i + self.offset)

    def map_rule(self, rule):
        head = [self.mapping[atom] for atom in rule[1]]
        body = [
            int(copysign(self.mapping[abs(atom)], atom)) for atom in rule[2]
        ]
        return (rule[0], head, body)

    def map_rules(self):
        self.rules = [self.map_rule(rule) for rule in self.rules]

    def map_weight_rule(self, rule, simplify=False):
        head = [self.mapping[atom] for atom in rule[1]]
        body = [(int(copysign(self.mapping[abs(a)], a)), w) for a, w in rule[3]]
        if not simplify:
            return (rule[0], head, rule[2], body)
        else:
            return (rule[0], head, rule[2][0], body)

    def map_weight_rules(self):
        self.weight_rules = [
            self.map_weight_rule(rule) for rule in self.weight_rules
        ]

    def handle_externals(self):
        for symbol, literal in self.primed_externals.items():
            self.primed_externals[symbol] = self.mapping[literal]
        for symbol, literal in self.normal_externals.items():
            self.normal_externals[symbol] = self.mapping[literal] - self.offset
            self.rules.append((True, [self.mapping[literal]], []))

    def set_output(self):
        # gather output if needed
        if self.solve_for_output:
            with self.ctl.solve(yield_=True) as handle:
                for m in handle:
                    break
                if handle.get().unsatisfiable:
                    raise Exception(STR_UNSAT)
        # map
        idx = 0
        for atom, symbol in self.output:
            mapped_atom = self.mapping[atom]
            if atom == 0 or mapped_atom == TRUE:
                self.output_facts.append(str(symbol))
                continue
            elif mapped_atom == FALSE:
                continue
            elif mapped_atom > self.offset:
                mapped_atom -= self.offset
            else:                                          # primed external
                symbol = clingo.Function(symbol.name[:-1], symbol.arguments)
            self.output[idx] = (mapped_atom, str(symbol))
            idx += 1
        self.output = self.output[0:idx]

    def do_set_init(self):
        self.init = [self.mapping[
                         self.ctl.symbolic_atoms[
                             clingo.Function(i.symbol.arguments[0].name+"'",
                                             i.symbol.arguments[0].arguments
                                            )
                         ].literal
                     ] for i in self.ctl.symbolic_atoms if i.symbol.name == INIT
        ]

    def set_init(self):
        #try:
        self.do_set_init()
        #except Exception:
        #    raise Exception(INIT_INCORRECT)

    #
    # observer
    #

    def update_offset(self, _list):
        for i in _list:
            if abs(i)>self.offset:
                self.offset = abs(i)
    
    def rule(self, choice, head, body):
        self.update_offset(head)
        self.update_offset(body)
        self.rules.append((choice, head, body))
   
    def weight_rule(self, choice, head, lower_bound, body):
        self.update_offset(head)
        self.update_offset([x for x,y in body])
        self.weight_rules.append((choice, head, lower_bound, body))
    
    def external(self, atom, value):
        self.update_offset([atom])
   
    def output_atom(self, symbol, atom):
        self.output.append((atom, symbol))

    #
    # __str__
    #
    
    def __str__(self):
        out = ""
        out += "\nOFFSET\n" + str(self.offset)
        out += "\nMAPPING\n" + "\n".join(
            ["{}:{}".format(i, item)
             for i, item in enumerate(self.mapping)]
        )
        out += "\nRULES\n" + "\n".join(
            ["{}:{}:{}".format(
                i[0], i[1], i[2]
            ) for i in self.rules if i is not None]
        )
        out += "\nWEIGHT RULES\n" + "\n".join(
            ["{}:{}:{}:{}".format(
                i[0], i[1], i[2], i[3]
            ) for i in self.weight_rules if i is not None]
        )
        out += "\nPRIMED EXTERNALS\n" + "\n".join(
            ["{}:{}".format(x, y) for x, y in sorted(
                self.primed_externals.items()
            )]
        )
        out += "\nNORMAL EXTERNALS\n" + "\n".join(
            ["{}:{}".format(x, y) for x, y in sorted(
                self.normal_externals.items()
            )]
        )
        out += "\nOUTPUT\n" + "\n".join(
            ["{}:{}".format(x, y) for x, y in sorted(self.output)]
        )
        out += "\nOUTPUT FACTS\n" + "\n".join(
            ["{}".format(x) for x in sorted(self.output_facts)]
        )
        out += "\nINIT\n" + "\n".join(
            ["{}".format(x) for x in sorted(self.init)]
        )
        return out

class Symbol:
    
    def __init__(self, symbol, is_fact, is_external, literal):
        # set up data that is found on a symbol
        self.symbol = symbol
        self.is_fact = is_fact
        self.is_external = is_external
        self.literal = literal

class SymbAtoms:
    # simulate the SymbolicAtoms class from clingo

    def __init__(self):
        self.symbols = {}

    def __getitem__(self,y):
        return self.symbols[y]

    def __iter__(self):
        for x, v in self.symbols.iteritems():
            yield v

    def add_term(self, term, is_fact, is_ext, lit):
        # pass a clingo term
        # also pass the relevant symbol information
        # add the clingo term as the key and the symbol as the value
        s = Symbol(term, is_fact, is_ext, lit)

        self.symbols[term] = s
        

class FakeControl:
    # simulate a clingo control object

    def __init__(self, files, adds="", options=[]):
        self.files = files
        self.adds = adds
        self.options = options

        # simulate symbolic atoms list
        self.symbolic_atoms = SymbAtoms()

        self.observers = []

        self.external_lits = []

        self.output_atoms = []

    def register_observer(self, observer):
        self.observers.append(observer)

    def ground(self):
        # add extra program to temp file if needed
        tmp = "temp12345.lp"

        if self.adds != "":
            with open(tmp, "w") as f:
                f.write(self.adds)
            
            self.files += [tmp]

        # subprocess.check_output throws error with non zero return value.
        # clingo returns 21(or something similar) for succesful grounding
        # so it will always be an error, hence the try block
        try:
            self.ground_rules = subprocess.check_output(["clingo"] + self.files + ["--pre"] + self.options)
        except subprocess.CalledProcessError as e:
            # becase its always an error, grab the "error" as the output
            # which should be the aspif program
            self.ground_rules = e.output

        if self.adds != "":
            # delete temp if it was created
            subprocess.check_call(["rm", tmp])

        #print(self.ground_rules)

    def parse(self):
        # use del to skip the first line saying the aspif version and the last empty line
        rules = self.ground_rules.split("\n")
        del rules[0]
        del rules[-1]
        for line in rules:
            line = line.split()

            if line[0] == "1":
                self.parse_rule(line)
            if line[0] == "5":
                self.parse_external(line)
            if line[0] == "4":
                self.parse_output(line)

        self.set_output()

    def parse_output(self, line):

        # curate line
        # sometimes predicate name has a space in it and it splits into multiple parts separated by space -> error with .split()
        # here we check that the length of the string is equal to the byte count
        # if it isnt then we merge the two strings into one and create a new "line"
        # repeat as needed

        while len(line[2]) != int(line[1]):
            new_line = line[:2]
            new_line.append(" ".join(line[2:4]))
            new_line += line[4:]
            line = new_line
            
        
        is_fact = line[3] == "0"

        term = clingo.parse_term(line[2])

        if is_fact:
            is_ext = False
            # lit is None because we can't know which literal belongs to this fact
            # the literal for a fact is also never needed in the generator class
            # so this should be fine
            # unless we want to output facts :/
            lit = None
            
            self.symbolic_atoms.add_term(term, is_fact, is_ext, lit)

        else:
            lit = long(line[-1])
            is_ext = lit in self.external_lits

            self.symbolic_atoms.add_term(term, is_fact, is_ext, lit)


        # if the term has the name "show" then we add the arguments to the output
        # to be processed later
        #term = clingo.parse_term(line[2])
        if term.name == "show":
            name = str(term.arguments[0])
            arity = int(term.arguments[1].number)

            self.output_atoms.append((name, arity))




    def set_output(self):

        # for each output found in the rules search for a matching symbol
        # this won't work for facts since facts have no literals
        for name, arity in self.output_atoms:
            for symbol in self.symbolic_atoms:
                if (symbol.symbol.name == name and
                     len(symbol.symbol.arguments) == arity):
                    for obs in self.observers:
                        obs.output_atom(symbol.symbol, symbol.literal)


    def parse_external(self, line):
        lit = long(line[1])
        val = int(line[2])

        self.external_lits.append(lit)
            
        for obs in self.observers:
            obs.external(lit, val)

    def parse_rule(self, line):
        
        # here we need to be careful because the values in the "line" are strings
        # but sometimes we need them as ints!

        is_choice = line[1] == "1"
        head_elements = int(line[2])
        if head_elements == 0:
            # constraint
            head = []
        else:
            # get all heads and convert them to longs in case the number is big
            head = map(long, list(line[3:3+head_elements]))

        # head elements influences the position of the body type marker
        # and also of the start of the body atoms. So we "shift" their positions
        # when grabbing from the list
        body_type = int(line[3 + head_elements])

        if body_type == 0:
            #normal body
            body_elements = line[4 + head_elements]
            if body_elements == "0":
                body = []
            else:
                body = map(long, list(line[5 + head_elements:]))
    
            for obs in self.observers:
                obs.rule(is_choice, head, body)

        if body_type == 1:
            #weight body
            lower_bound = int(line[4 + head_elements])
            number_of_lits = int(line[5 + head_elements])
            body = map(long, list(line[6 + head_elements:]))
            body = [body[i:i+2] for i in xrange(0,len(body),2)]
    
            for obs in self.observers:
                obs.weight_rule(is_choice, head, lower_bound, body)

class DLPGeneratorClingoPre(DLPGenerator):
    # pretty much equal to DLP generator, just slightly different arguments
    # and some small differences in the set_externals and set_output functions

    def __init__(self, files = [], adds = "", options = []):
        # input
        self.files = files
        self.adds = adds
        self.options = options
        # output
        self.offset = 0
        self.rules = []
        self.weight_rules = []
        self.primed_externals = {}
        self.normal_externals = {}
        self.output = []
        self.output_facts = []
        self.init = []
        # rest
        self.ctl = None
        self.next = {}
        self.mapping = []
        self.solve_for_output = True

    def run(self):
        # preliminaries
        self.ctl = FakeControl(self.files, self.adds, self.options)
        self.ctl.register_observer(self)
        self.ctl.ground()
        self.ctl.parse()
        #print(self)
        # analyze
        self.set_externals()
        self.simplify()
        self.set_next()
        self.set_mapping()
        self.map_rules()
        self.map_weight_rules()
        self.handle_externals()
        self.set_output()
        self.set_init()
        # return
        return DynamicLogicProgramContainer(
            self.offset, self.rules, self.weight_rules,
            self.primed_externals, self.normal_externals,
            self.output, self.output_facts, self.init
        )

    def set_externals(self):
        for x in self.ctl.symbolic_atoms:
            if x.is_external:
                if len(x.symbol.name) and x.symbol.name[-1]=="'":
                    self.primed_externals[x.symbol] = x.literal
                else:
                    self.normal_externals[x.symbol] = x.literal

    def set_output(self):
        # map
        # delete the solve if havent solved before that is in the backend DLP Generator
        idx = 0
        for atom, symbol in self.output:
            mapped_atom = self.mapping[atom]
            if atom == 0 or mapped_atom == TRUE:
                self.output_facts.append(str(symbol))
                continue
            elif mapped_atom == FALSE:
                continue
            elif mapped_atom > self.offset:
                mapped_atom -= self.offset
            else:                                          # primed external
                symbol = clingo.Function(symbol.name[:-1], symbol.arguments)
            self.output[idx] = (mapped_atom, str(symbol))
            idx += 1
        self.output = self.output[0:idx]


SAtom  = namedtuple('SAtom', 
                    [ 'heads',  'bodypos',  'bodyneg',
                     'wheads', 'wbodypos', 'wbodyneg'])
# heads, bodypos and bodyneg are lists of rule positions,
# wheads is a list of weight_wrule positions,
# and wbodypos and wbodyneg are tuples (weight, weight_rule position)

class DLPGeneratorSimplifier(DLPGenerator):

    def __init__(
        self, files = [], adds = [], parts = [], options = [],
        compute_cautious=True, compute_brave=True):
        # input
        DLPGenerator.__init__(self, files, adds, parts, options)
        self.compute_cautious = compute_cautious
        self.compute_brave = compute_brave
        # rest
        self.satoms = []
        self.true = []
        self.false = []
        self.cautious = []
        self.add_constraints = []
        self.solve_for_output = False
        
        # maximum time that calculating cautious or brave consequences can take
        self.time_limit = 0.001

    def simplify(self):
        self.mapping = [None]*len(self.satoms)
        self.offset = len(self.satoms) - 1
        if self.compute_brave:
            print("false before brave: {}".format(len(self.false)))
            self.false += self.get_consequences("brave", False)
            print("false after brave: {}".format(len(self.false)))
        if self.compute_cautious:
            self.cautious += self.get_consequences("cautious", True)

        self.fitting()
        print("DLP: Rules before simplifying: {}".format(len(self.rules)))

        if not self.compute_brave and not self.compute_cautious:
            self.solve_for_output = True

    def get_consequences(self, opt, true):
        self.ctl.configuration.solve.enum_mode = opt

        with self.ctl.solve(yield_=True, async=True) as handle:
            time_used = 0
            last = None
            while time_used < self.time_limit and handle.wait(self.time_limit - time_used):
                t = time()
                try:
                    last = handle.next()
                    print(len(last.symbols(shown=True)))
                except StopIteration:
                    time_used += time() - t
                    break

                time_used += time() - t
             
            print("DLP: Time used on {} consequences: {}".format(opt, time_used))

            if last is None:
                # if we found nothing in the alloted time then return nothing
                return []
   
            elif true:
                symbols = last.symbols(shown=True)
            else:
                symbols = last.symbols(shown=True, complement=True)

        return [self.ctl.symbolic_atoms[x].literal for x in symbols]

    def remove_rule_from_heads(self, rule, atom, weight=False):
        satom = self.satoms[atom]
        if weight:
            satom.wheads.remove(rule)
        else:
            satom.heads.remove(rule)
        if not satom.heads and not satom.wheads:
            self.false.append(atom)

    def rule_is_single_constraint(self, idx, rule, tmp_cautious, weight=False):
        if weight:
            literal, _ = rule[3].pop()
            self.weight_rules[idx] = None
        else:
            literal = rule[2].pop()
            self.rules[idx] = None
        if literal > 0:
            self.false.append(literal)
        else:
            tmp_cautious.append(-literal)

    def fitting(self):
        # preprocess true facts and cautious
        tmp_cautious = [x for x in self.cautious if self.satoms[x] is not None]
        self.cautious = set()
        self.true = [x for x in self.true if self.satoms[x] is not None]
        while True:
            # preprocessing
            if len(self.true):
                atom = self.true[0]
                self.true = self.true[1:]
                if self.mapping[atom] == TRUE:
                    continue
                else:
                    self.mapping[atom] = TRUE
                atom_type = FITTING_TRUE
                signed = atom
                self.offset -= 1
            elif len(self.false):
                atom = self.false[0]
                self.false = self.false[1:]
                if self.mapping[atom] == FALSE:
                    continue
                else:
                    self.mapping[atom] = FALSE
                atom_type = FITTING_FALSE
                signed = -atom
                self.offset -= 1
            elif len(tmp_cautious):
                atom = tmp_cautious[0]
                tmp_cautious = tmp_cautious[1:]
                if self.mapping[atom] == TRUE or (atom in self.cautious):
                    continue
                self.cautious.add(atom)
                atom_type = FITTING_CAUTIOUS
                signed = atom
            else:
                return
            satom = self.satoms[atom]
            if satom is None: # fact not appearing elsewhere
                continue
            # selection
            if atom_type == FITTING_TRUE:
                heads             = satom.heads
                wheads            = satom.wheads
                satisfied_body    = satom.bodypos
                unsatisfied_body  = satom.bodyneg
                satisfied_wbody   = satom.wbodypos
                unsatisfied_wbody = satom.wbodyneg
            elif atom_type == FITTING_FALSE:
                heads             = satom.heads
                wheads            = satom.wheads
                satisfied_body    = satom.bodyneg
                unsatisfied_body  = satom.bodypos
                satisfied_wbody   = satom.wbodyneg
                unsatisfied_wbody = satom.wbodypos
            else: # FITTING_CAUTIOUS
                heads             = []
                wheads            = []
                satisfied_body    = []
                unsatisfied_body  = satom.bodyneg
                satisfied_wbody   = []
                unsatisfied_wbody = satom.wbodyneg
            # heads
            for i in heads:
                rule = self.rules[i]
                if rule is None:
                    continue
                if not rule[0]: # disjunction
                    if atom_type == FITTING_TRUE:
                        for head in rule[1]:
                            if head != atom:
                                self.remove_rule_from_heads(i, head)
                        self.rules[i] = None
                    elif atom_type == FITTING_FALSE:
                        rule[1].remove(atom)
                        if len(rule[1]) == 1 and not rule[2]:
                            self.true.append(rule[1].pop())     # fact
                            self.rules[i] = None
                        elif not rule[1] and len(rule[2]) == 1: # 1-constraint
                            self.rule_is_single_constraint(
                                i, rule, tmp_cautious
                            )
                else:           # choice
                    rule[1].remove(atom)
                    if not rule[1]:
                        self.rules[i] = None
            # wheads
            for i in wheads:
                rule = self.weight_rules[i]
                if rule is None:
                    continue
                if not rule[0]: # disjunction
                    if atom_type == FITTING_TRUE:
                        for head in rule[1]:
                            if head != atom:
                                self.remove_rule_from_heads(i, head, True)
                        self.weight_rules[i] = None
                    elif atom_type == FITTING_FALSE:
                        rule[1].remove(atom)
                        # rule[2] == [0] if body is SAT
                        if len(rule[1]) == 1 and rule[2]==[0]: # fact
                            self.true.append(rule[1].pop())
                            self.weight_rules[i] = None
                        elif not rule[1] and len(rule[3]) == 1: # 1-constraint
                            self.rule_is_single_constraint(
                                i, rule, tmp_cautious, True
                            )
                else:            # choice
                    rule[1].remove(atom)
                    if not rule[1]:
                        self.weight_rules[i] = None
            # satisfied body
            for i in satisfied_body:
                rule = self.rules[i]
                if rule is None:
                    continue
                rule[2].remove(signed)
                if not rule[0] and len(rule[1]) == 1 and not rule[2]: # fact
                    self.true.append(rule[1].pop())
                    self.rules[i] = None
                elif not rule[0] and not rule[1] and len(rule[2]) == 1: # 1-c
                    self.rule_is_single_constraint(i, rule, tmp_cautious)
            # unsatisfied body
            for i in unsatisfied_body:
                rule = self.rules[i]
                if rule is None:
                    continue
                for head in rule[1]:
                    self.remove_rule_from_heads(i, head)
                self.rules[i] = None
            # satisfied wbody
            for i, w in satisfied_wbody:
                rule = self.weight_rules[i]
                if rule is None:
                    continue
                rule[3].remove((signed, w))
                rule[2][0] -= w
                if rule[2][0] <= 0: # body is SAT
                    rule[2][0] = 0
                    if not rule[0] and len(rule[1]) == 1: # fact
                        self.true.append(rule[1].pop())
                        self.weight_rules[i] = None
                elif not rule[0] and not rule[1] and len(rule[3]) == 1: # 1-c
                    self.rule_is_single_constraint(i, rule, tmp_cautious, True)
            # unsatisfied wbody
            for i, w in unsatisfied_wbody:
                rule = self.weight_rules[i]
                if rule is None:
                    continue
                rule[3].remove((-signed, w))
                if sum([ww for _, ww in rule[3]]) < rule[2][0]: # body is UNSAT
                    for head in rule[1]:
                        self.remove_rule_from_heads(i, head, True)
                    self.weight_rules[i] = None
                elif not rule[0] and not rule[1] and len(rule[3]) == 1: # 1-c
                    self.rule_is_single_constraint(i, rule, tmp_cautious, True)
            # finish
            if atom_type == FITTING_CAUTIOUS:
                self.satoms[atom] = SAtom(
                    satom.heads, satom.bodypos, [],
                    satom.wheads, satom.wbodypos, []
                )
            else:
                self.satoms[atom] = None

    def set_next(self):
        for symbol, literal in self.primed_externals.items():
            # handle FALSE primed externals
            if self.mapping[literal] == FALSE:
                self.offset += 1
                self.rules.append((False, [], [literal]))
            self.mapping[literal] = PRIMED_EXTERNAL
            next_literal = self.get_next_literal(symbol, 0)
            self.next[literal] = next_literal
            if next_literal and self.mapping[next_literal] not in {TRUE, FALSE}:
                self.offset -= 1

    def set_mapping(self):
        # handle FALSE normal externals
        for _, literal in self.normal_externals.items():
            if self.mapping[literal] == FALSE:
                self.offset += 1
                self.rules.append((False, [], [literal]))
                self.mapping[literal] = None
        # do atoms and normal externals
        number = -1           # we do self.mapping[0]=offset
        for idx, item in enumerate(self.mapping):
            if item is None:
                number += 1
                self.mapping[idx] = number + self.offset
        # do primed externals
        for symbol, literal in self.primed_externals.items():
            next_literal = self.next[literal]
            next_mapping = self.mapping[next_literal]
            if not next_literal or next_mapping == FALSE:
                number += 1
                self.mapping[literal] = number
                self.add_constraints.append( number + self.offset)
            elif next_mapping == TRUE:
                number += 1
                self.mapping[literal] = number
                self.add_constraints.append(-number - self.offset)
            else:
                self.mapping[literal] = next_mapping - self.offset
        assert number == self.offset

    def map_rules(self):
        self.rules = [
            self.map_rule(rule) for rule in self.rules if rule is not None
        ] + [
            (False, [], [-self.mapping[i]]) for i in self.cautious
                if self.mapping[i] != TRUE
        ] + [
            (False, [], [i]) for i in self.add_constraints
        ]

    def map_weight_rules(self):
        self.weight_rules = [
            self.map_weight_rule(rule, True) for rule in self.weight_rules
                if rule is not None
        ]

    #
    # observer
    #

    def add_satoms(self, i, start=True):
        if i >= len(self.satoms):
            self.satoms += [None] * (i-len(self.satoms)+1)
        if start and self.satoms[i] == None:
            self.satoms[i] = SAtom(set(), [], [], set(), [], [])

    def rule(self, choice, head, body):
        # facts
        if not choice and len(head) == 1 and not len(body):
            self.add_satoms(head[0], False)
            self.true.append(head[0])
        # 1-constraints
        elif not len(head) and len(body) == 1:
            literal = body[0]
            self.add_satoms(abs(literal))
            if literal >= 0:
                self.false.append(literal)
            else:
                self.cautious.append(-literal)
        # rest
        else:
            rule = len(self.rules)
            for i in head:
                self.add_satoms(i)
                self.satoms[i].heads.add(rule)
            for i in body:
                self.add_satoms(abs(i))
                if i > 0:
                    self.satoms[i].bodypos.append(rule)
                else:
                    self.satoms[-i].bodyneg.append(rule)
            self.rules.append((choice, set(head), set(body)))

    def weight_rule(self, choice, head, lower_bound, body):
        # no body
        if not len(body) or lower_bound <= 0:
            self.rule(choice, head, [])
        # lower_bound unreachable
        elif sum([w for _, w in body]) < lower_bound:
            pass
        # singleton body
        elif len(body) == 1:
            self.rule(choice, head, [body[0][0]])
        # rest
        else:
            wrule = len(self.weight_rules)
            for i in head:
                self.add_satoms(i)
                self.satoms[i].wheads.add(wrule)
            for i, w in body:
                self.add_satoms(abs(i))
                if i > 0:
                    self.satoms[i].wbodypos.append((wrule,w))
                else:
                    self.satoms[-i].wbodyneg.append((wrule,w))
            self.weight_rules.append(
                (choice, set(head), [lower_bound], set(body))
            )

    def external(self, atom, value):
        self.add_satoms(atom, False)

class DynamicLogicProgramContainer:
    # just a container for the data

    def __init__(self, offset, rules, weight_rules,
                 primed_externals, normal_externals,
                 output, output_facts, init):
        # init
        self.offset = offset
        self.rules = rules
        self.weight_rules = weight_rules
        self.primed_externals = primed_externals
        self.normal_externals = normal_externals
        self.output = output
        self.output_facts = output_facts
        self.init = init


time_str = "%times for transition \n#program base. \n time(1000000001)."

time_0 = "1000000000"
time_1 = "1000000001"

class Grounder(object):

    def __init__(self, files, program="", options=[], debug=False):

        self.debug = debug
        self.debug_gf_name = "g.lp"
        # write program + time to tempfile
        self.prog_file = NamedTemporaryFile(delete=False)

        self.prog_file.write(program)
        self.prog_file.write(time_str)
        self.prog_file.flush()

        files += [self.prog_file.name]

        # ground file as temp file
        self.gf = NamedTemporaryFile(delete=False)

        self.ground_command = "clingo {} {} --text > {}".format(" ".join(files), " ".join(options), self.gf.name)

        self.grounded_rules = []
        self.other_rules = []
        self.init = []

        self.timelimit = None

        self.has_grounded = False

        self.grounding_time = 0


    def parse_prog(self):

        if self.has_grounded:
            print("The grounding process has already been done, skipping...")
            return

        start_time = time()

        subprocess.call(self.ground_command, shell=True)
        self.prog_file.close()
        os.unlink(self.prog_file.name)

        main_criteria = [time_0, time_1]
        other_rule_criteria = ["timelimit", "#show"]

        with open(self.gf.name, "r") as f:
            for rule in f.readlines():
                if rule.strip() == "time(" + time_0 + ")." or rule.strip() == "time(" + time_1 + ").":
                    continue
                elif any(check in rule for check in other_rule_criteria):
                    if "timelimit(" in rule:
                        self.timelimit = int(rule[rule.find("(") + 1:rule.rfind(")")])
                    else:
                        self.other_rules.append(rule)
                elif "#external" in rule and time_1 in rule:
                    self.grounded_rules.append(rule)
                elif "init(" in rule:
                    atom = rule[rule.find("(")+1:rule.rfind(")")].replace(time_0, "0")
                    self.init.append(atom)
                elif "#external" not in rule and any(check in rule for check in main_criteria):
                    self.grounded_rules.append(rule)

        self.has_grounded = True

        if self.debug:
            with open(self.gf.name, "r") as f:
                with open(self.debug_gf_name, "w") as df:
                    df.write(f.read())

        self.gf.close()
        os.unlink(self.gf.name)

        self.grounding_time += time() - start_time

    def ground_range(self, start, end):

        start_time = time()

        if not self.has_grounded:
            print("ERROR: grounding has not taken place yet")
            return

        rules = {}

        for t in range(start, end+1):
            rules[t] = [rule.replace(time_0, str(t - 1)).replace(time_1, str(t)) for rule in self.grounded_rules]

        self.grounding_time += time() - start_time

        return rules

    def ground(self, gtime):

        start_time = time()

        if not self.has_grounded:
            print("ERROR: grounding has not taken place yet")
            return

        rules = {}

        rules[gtime] = [rule.replace(time_0, str(time - 1)).replace(time_1, str(gtime)) for rule in self.grounded_rules]

        self.grounding_time += time() - start_time

        return rules

    def ground_to_file(self, start, end, filename="grounded.lp"):

        if not self.has_grounded:
            self.parse_prog()

        with open(filename, "w") as f:
            f.writelines(self.other_rules)
            f.writelines([init + ".\n" for init in self.init])
            for t, r in self.ground_range(start, end).iteritems():
                f.writelines(r)

    def __str__(self):
        string = ""
        for rule in self.other_rules:
            string += rule
        for rule in self.grounded_rules:
            string += rule

        return string

