#ifndef __PDDL_PARSE__DETAIL__PARSING__REQUIREMENT_H
#define __PDDL_PARSE__DETAIL__PARSING__REQUIREMENT_H

#include <pddlparse/ASTForward.h>
#include <pddlparse/Context.h>

namespace pddl
{
namespace detail
{

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Requirement
//
////////////////////////////////////////////////////////////////////////////////////////////////////

ast::Requirement parseRequirement(Context &context);
const char *toString(const ast::Requirement &requirement);

////////////////////////////////////////////////////////////////////////////////////////////////////

}
}

#endif
