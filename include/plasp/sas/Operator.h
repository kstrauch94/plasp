#ifndef __SAS__OPERATOR_H
#define __SAS__OPERATOR_H

#include <string>
#include <vector>

#include <plasp/sas/AssignedVariable.h>
#include <plasp/sas/Effect.h>
#include <plasp/sas/Predicate.h>
#include <plasp/sas/Variable.h>

namespace plasp
{
namespace sas
{

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Operator
//
////////////////////////////////////////////////////////////////////////////////////////////////////

class Operator;
using Operators = std::vector<Operator>;

////////////////////////////////////////////////////////////////////////////////////////////////////

class Operator
{
	public:
		static Operator fromSAS(std::istream &istream, const Variables &variables);

		using Condition = AssignedVariable;
		using Conditions = AssignedVariables;

	public:
		const Predicate &predicate() const;
		const Conditions &preconditions() const;
		const Effects &effects() const;
		size_t costs() const;

	private:
		Predicate m_predicate;
		Conditions m_preconditions;
		Effects m_effects;
		size_t m_costs;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}
}

#endif
