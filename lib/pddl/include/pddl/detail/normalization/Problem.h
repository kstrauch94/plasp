#ifndef __PDDL__DETAIL__NORMALIZATION__PROBLEM_H
#define __PDDL__DETAIL__NORMALIZATION__PROBLEM_H

#include <pddl/ASTForward.h>
#include <pddl/Context.h>
#include <pddl/NormalizedASTForward.h>

namespace pddl
{
namespace detail
{

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Problem
//
////////////////////////////////////////////////////////////////////////////////////////////////////

normalizedAST::ProblemPointer normalize(ast::ProblemPointer &&problem, normalizedAST::Domain *domain);

////////////////////////////////////////////////////////////////////////////////////////////////////

}
}

#endif
