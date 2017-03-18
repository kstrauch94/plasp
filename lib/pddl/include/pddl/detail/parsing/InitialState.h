#ifndef __PDDL__DETAIL__PARSING__INITIAL_STATE_H
#define __PDDL__DETAIL__PARSING__INITIAL_STATE_H

#include <pddl/Context.h>
#include <pddl/detail/ASTContext.h>

namespace pddl
{
namespace detail
{

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// InitialState
//
////////////////////////////////////////////////////////////////////////////////////////////////////

ast::InitialState parseInitialState(Context &context, ASTContext &astContext, VariableStack &variableStack);

////////////////////////////////////////////////////////////////////////////////////////////////////

}
}

#endif