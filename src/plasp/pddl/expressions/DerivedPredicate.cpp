#include <plasp/pddl/expressions/DerivedPredicate.h>

#include <plasp/pddl/Context.h>
#include <plasp/pddl/ExpressionContext.h>

namespace plasp
{
namespace pddl
{
namespace expressions
{

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// DerivedPredicate
//
////////////////////////////////////////////////////////////////////////////////////////////////////

DerivedPredicate::DerivedPredicate(size_t id)
:	m_id{id}
{
}

////////////////////////////////////////////////////////////////////////////////////////////////////

size_t DerivedPredicate::id() const
{
	return m_id;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void DerivedPredicate::setArgument(ExpressionPointer argument)
{
	m_argument = argument;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

ExpressionPointer DerivedPredicate::argument() const
{
	return m_argument;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void DerivedPredicate::print(std::ostream &ostream) const
{
	ostream << "(:derived <no name> ";
	m_argument->print(ostream);
	ostream << ")";
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}
}
}
