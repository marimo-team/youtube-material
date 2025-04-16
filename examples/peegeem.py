import marimo

__generated_with = "0.12.8"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    # Import necessary libraries
    import pandas as pd
    from pgmpy.models import DiscreteBayesianNetwork
    from pgmpy.estimators import MaximumLikelihoodEstimator
    from pgmpy.inference import VariableElimination
    from pgmpy.factors.discrete import DiscreteFactor # Import for type hinting
    import numpy as np
    from typing import Union, List, Any, Dict, Tuple # For type hinting
    import itertools # Needed for joint distribution output formatting
    import re # Needed for parsing state names for range queries

    # Forward declaration for type hinting
    class DAG: pass
    class Variable: pass
    class VariableCombination: pass
    class Condition: pass
    class QueryExpression: pass


    # --- Condition Class ---
    # (Remains the same)
    class Condition:
        """Represents a single condition on a variable (e.g., variable == value)."""
        def __init__(self, variable_name: str, operator: str, value: Any, dag_instance: DAG):
            self.variable_name = variable_name
            self.operator = operator
            self.value = value
            self._dag = dag_instance
        def __repr__(self): return f"Condition({self.variable_name} {self.operator} {repr(self.value)})"
        def __and__(self, other: Union['Condition', List['Condition']]) -> List['Condition']:
            if isinstance(other, Condition):
                if self._dag is not other._dag: raise ValueError("Cannot combine conditions from different DAG instances.")
                return [self, other]
            elif isinstance(other, list):
                if not all(isinstance(c, Condition) and c._dag is self._dag for c in other): raise ValueError("Cannot combine condition with list containing non-Conditions or Conditions from different DAGs.")
                return [self] + other
            else: return NotImplemented
        def __rand__(self, other: Union[Any]) -> List['Condition']:
             if isinstance(other, list):
                 if not all(isinstance(c, Condition) and c._dag is self._dag for c in other): raise ValueError("Cannot combine list with Condition from different DAGs or non-Condition.")
                 return other + [self]
             return NotImplemented

    # --- Variable Combination Class ---
    # (Remains the same)
    class VariableCombination:
        """Represents a combination of multiple variables for joint probability queries."""
        def __init__(self, variables: List[Variable]):
            if not variables: raise ValueError("VariableCombination cannot be empty.")
            first_dag = variables[0]._dag
            if not all(isinstance(v, Variable) and v._dag is first_dag for v in variables): raise ValueError("All items in VariableCombination must be Variables from the same DAG.")
            self.variables = list(dict.fromkeys(variables))
            self._dag = first_dag
        def __repr__(self): return f"VariableCombination([{', '.join(v.name for v in self.variables)}])"
        def __and__(self, other: Union[Variable, 'VariableCombination']) -> 'VariableCombination':
            if isinstance(other, Variable):
                if other._dag is not self._dag: raise ValueError("Cannot combine variables from different DAGs.")
                return VariableCombination(self.variables + [other])
            elif isinstance(other, VariableCombination):
                if other._dag is not self._dag: raise ValueError("Cannot combine variables from different DAGs.")
                return VariableCombination(self.variables + other.variables)
            return NotImplemented
        def __rand__(self, other: Variable) -> 'VariableCombination':
             if isinstance(other, Variable):
                 if other._dag is not self._dag: raise ValueError("Cannot combine variables from different DAGs.")
                 return VariableCombination([other] + self.variables)
             return NotImplemented
        def __or__(self, conditions: Union[Condition, List[Condition]]) -> 'QueryExpression': return QueryExpression(target=self, conditions=conditions)

    # --- Updated Variable Class ---
    # (Remains the same)
    class Variable:
        """Represents a variable (node) in the DAG with overloaded operators. Made hashable."""
        def __init__(self, name: str, dag_instance: DAG):
            self.name = name
            self._dag = dag_instance
        def __repr__(self): return f"Variable({self.name})"
        def __str__(self): return self.name
        def __eq__(self, other: object) -> Union[bool, Condition]:
            if isinstance(other, Variable): return self.name == other.name and self._dag is other._dag
            else:
                processed_value, _ = self._process_value_for_state_check(other)
                return Condition(self.name, '==', processed_value, self._dag)
        def __ne__(self, other: object) -> Condition:
            if isinstance(other, Variable): raise TypeError("Cannot use '!=' between two Variable objects.")
            processed_value, _ = self._process_value_for_state_check(other)
            return Condition(self.name, '!=', processed_value, self._dag)
        def __gt__(self, other: object) -> Condition:
            if isinstance(other, Variable): raise TypeError("Cannot use '>' between two Variable objects.")
            processed_value, is_numeric = self._process_value_for_state_check(other)
            if not is_numeric: print(f"Warning: Applying '>' to non-numeric value '{repr(other)}' for variable '{self.name}'.")
            return Condition(self.name, '>', processed_value, self._dag)
        def __ge__(self, other: object) -> Condition:
            if isinstance(other, Variable): raise TypeError("Cannot use '>=' between two Variable objects.")
            processed_value, is_numeric = self._process_value_for_state_check(other)
            if not is_numeric: print(f"Warning: Applying '>=' to non-numeric value '{repr(other)}' for variable '{self.name}'.")
            return Condition(self.name, '>=', processed_value, self._dag)
        def __lt__(self, other: object) -> Condition:
            if isinstance(other, Variable): raise TypeError("Cannot use '<' between two Variable objects.")
            processed_value, is_numeric = self._process_value_for_state_check(other)
            if not is_numeric: print(f"Warning: Applying '<' to non-numeric value '{repr(other)}' for variable '{self.name}'.")
            return Condition(self.name, '<', processed_value, self._dag)
        def __le__(self, other: object) -> Condition:
            if isinstance(other, Variable): raise TypeError("Cannot use '<=' between two Variable objects.")
            processed_value, is_numeric = self._process_value_for_state_check(other)
            if not is_numeric: print(f"Warning: Applying '<=' to non-numeric value '{repr(other)}' for variable '{self.name}'.")
            return Condition(self.name, '<=', processed_value, self._dag)
        def _process_value_for_state_check(self, value: Any) -> Tuple[Any, bool]:
            is_numeric = isinstance(value, (int, float))
            try:
                if not hasattr(self._dag, 'model') or not hasattr(self._dag, '_bool_cols'): raise AttributeError("DAG model not fully initialized.")
                processed_value = str(value) if self.name in self._dag._bool_cols else value
                return processed_value, is_numeric
            except Exception as e:
                print(f"Warning: Could not fully process/verify value '{repr(value)}' for variable '{self.name}': {e}")
                return value, is_numeric
        def __or__(self, conditions: Union[Condition, List[Condition]]) -> 'QueryExpression': return QueryExpression(target=self, conditions=conditions)
        def __and__(self, other: Union[Variable, VariableCombination]) -> VariableCombination:
            if isinstance(other, Variable):
                if other._dag is not self._dag: raise ValueError("Cannot combine variables from different DAGs.")
                return VariableCombination([self, other])
            elif isinstance(other, VariableCombination): return other.__rand__(self)
            return NotImplemented
        def __hash__(self) -> int: return hash(self.name)

    # --- Updated Query Expression Class ---
    # (Remains the same)
    class QueryExpression:
        """Represents a probability query expression."""
        def __init__(self, target: Union[Variable, VariableCombination], conditions: Union[Condition, List[Condition]]):
            self.target = target
            self._dag = target._dag
            if isinstance(conditions, Condition): self.conditions = [conditions]
            elif isinstance(conditions, list): self.conditions = conditions
            else: raise TypeError(f"Conditions must be a Condition or list of Conditions, got {type(conditions)}")
            if not all(c._dag is self._dag for c in self.conditions): raise ValueError("Target variable(s) and conditions must belong to the same DAG instance.")
        @property
        def target_names(self) -> List[str]:
            if isinstance(self.target, Variable): return [self.target.name]
            elif isinstance(self.target, VariableCombination): return [v.name for v in self.target.variables]
            else: raise TypeError("Invalid target type in QueryExpression")
        def __repr__(self):
            target_repr = self.target.name if isinstance(self.target, Variable) else repr(self.target)
            return f"QueryExpression(Target: {target_repr}, Conditions: {self.conditions})"

    # --- Helper Function for Range Queries ---
    # (Remains the same)
    def _get_matching_states(variable_name: str, operator: str, value: Any, dag_instance: DAG) -> List[Any]:
        """Identifies discrete states of a variable that satisfy a given condition."""
        try:
            cpd = dag_instance.model.get_cpds(variable_name)
            if not cpd: return []
            all_states = cpd.state_names[variable_name]
        except Exception as e:
            print(f"Warning: Could not retrieve states for variable '{variable_name}' for range query: {e}")
            return []
        matching_states = []
        value_is_numeric = isinstance(value, (int, float))
        for state in all_states:
            state_matches = False
            try:
                state_numeric = None; range_match = re.match(r"([-+]?\d*\.?\d+)\s*-\s*([-+]?\d*\.?\d+)", str(state))
                try: state_numeric = float(state); is_single_number_state = True
                except ValueError: is_single_number_state = False
                if value_is_numeric:
                    if range_match:
                        parsed_lower = float(range_match.group(1)); is_plus_range = range_match.group(2) == '+'
                        parsed_upper = float('inf') if is_plus_range else float(range_match.group(2))
                        if operator == '>': state_matches = parsed_lower > value
                        elif operator == '>=': state_matches = parsed_lower >= value
                        elif operator == '<': state_matches = parsed_upper < value
                        elif operator == '<=': state_matches = parsed_upper <= value
                        elif operator == '==': state_matches = parsed_lower <= value < parsed_upper
                        elif operator == '!=': state_matches = not (parsed_lower <= value < parsed_upper)
                    elif is_single_number_state:
                        if operator == '>': state_matches = state_numeric > value
                        elif operator == '>=': state_matches = state_numeric >= value
                        elif operator == '<': state_matches = state_numeric < value
                        elif operator == '<=': state_matches = state_numeric <= value
                        elif operator == '==': state_matches = np.isclose(state_numeric, value)
                        elif operator == '!=': state_matches = not np.isclose(state_numeric, value)
                    else: # Cannot compare numerically with non-numeric state
                         if isinstance(value, str):
                             if operator == '==': state_matches = (state == value)
                             elif operator == '!=': state_matches = (state != value)
                else: # Value is not numeric
                     if operator == '==': state_matches = (str(state) == str(value))
                     elif operator == '!=': state_matches = (str(state) != str(value))
            except Exception as parse_error:
                print(f"Warning: Could not parse state '{state}' for comparison in variable '{variable_name}': {parse_error}")
                state_matches = False
            if state_matches: matching_states.append(state)
        if not matching_states: print(f"Warning: Condition '{variable_name} {operator} {repr(value)}' did not match any discrete states.")
        return matching_states

    # --- Updated P_Calculator Class (Standalone) ---
    # (Remains the same)
    class P_Calculator:
        """Callable class to calculate probabilities (marginal or joint)."""
        def __call__(self, query_input: Union[Variable, VariableCombination, QueryExpression]) -> List[Dict[str, Any]]:
            target_obj: Union[Variable, VariableCombination]; all_conditions: List[Condition] = []; dag_instance: DAG
            if isinstance(query_input, (Variable, VariableCombination)):
                target_obj = query_input; dag_instance = query_input._dag; all_conditions = []
            elif isinstance(query_input, QueryExpression):
                target_obj = query_input.target; dag_instance = query_input._dag; all_conditions = query_input.conditions
            else: raise TypeError("Input to P() must be a Variable, VariableCombination, or QueryExpression.")
            equality_evidence = {}; range_conditions = []; query_condition_reprs = []
            for cond in all_conditions:
                query_condition_reprs.append(repr(cond))
                if cond.operator == '==':
                     if cond.variable_name in equality_evidence and equality_evidence[cond.variable_name] != cond.value: raise ValueError(f"Conflicting equality evidence for variable '{cond.variable_name}'.")
                     equality_evidence[cond.variable_name] = cond.value
                elif cond.operator in ['!=', '>', '>=', '<', '<=']: range_conditions.append(cond)
                else: raise NotImplementedError(f"Condition operator '{cond.operator}' is not yet supported.")
            if isinstance(target_obj, Variable): target_names = [target_obj.name]
            else: target_names = [v.name for v in target_obj.variables]
            range_vars = list(set(rc.variable_name for rc in range_conditions)); query_vars = list(dict.fromkeys(target_names + range_vars))
            target_repr = ', '.join(target_names); condition_repr = ', '.join(query_condition_reprs) if query_condition_reprs else "None"
            query_repr = f"P({target_repr} | {condition_repr})"; print(f"Calculating {query_repr} using standalone P_Calculator")
            if range_conditions: print(f"  (Handling range conditions: {[repr(rc) for rc in range_conditions]})")
            try:
                joint_factor: DiscreteFactor = dag_instance.inference.query(variables=query_vars, evidence=equality_evidence if equality_evidence else None, show_progress=False)
                if range_conditions:
                    print(f"  Applying range conditions by filtering factor over {joint_factor.variables}...")
                    filtered_factor = joint_factor.copy(); factor_vars_ordered = filtered_factor.variables
                    state_name_map = filtered_factor.state_names; state_combinations = [state_name_map[var] for var in factor_vars_ordered]
                    flat_probabilities = filtered_factor.values.flatten(); new_probabilities = flat_probabilities.copy()
                    prob_idx = 0
                    for state_tuple in itertools.product(*state_combinations):
                        state_dict = dict(zip(factor_vars_ordered, state_tuple)); include_this_combination = True
                        for rc in range_conditions:
                            var, op, val, current_state = rc.variable_name, rc.operator, rc.value, state_dict[rc.variable_name]
                            state_matches_condition = False
                            try: # Simplified check logic (should match helper function)
                                state_numeric = None; value_is_numeric = isinstance(val, (int, float))
                                try: state_numeric = float(current_state)
                                except ValueError: pass
                                if value_is_numeric and state_numeric is not None:
                                    if op == '>': state_matches_condition = state_numeric > val
                                    elif op == '>=': state_matches_condition = state_numeric >= val
                                    elif op == '<': state_matches_condition = state_numeric < val
                                    elif op == '<=': state_matches_condition = state_numeric <= val
                                    elif op == '!=': state_matches_condition = not np.isclose(state_numeric, val)
                                    elif op == '==': state_matches_condition = np.isclose(state_numeric, val)
                                elif isinstance(val, str) and state_numeric is None:
                                    if op == '!=': state_matches_condition = (str(current_state) != str(val))
                                    elif op == '==': state_matches_condition = (str(current_state) == str(val))
                            except Exception: pass
                            if not state_matches_condition: include_this_combination = False; break
                        if not include_this_combination: new_probabilities[prob_idx] = 0.0
                        prob_idx += 1
                    filtered_factor.values = new_probabilities.reshape(filtered_factor.cardinality)
                    vars_to_marginalize = [rv for rv in range_vars if rv not in target_names]
                    if vars_to_marginalize:
                        print(f"  Marginalizing out range variables: {vars_to_marginalize}")
                        final_factor = filtered_factor.marginalize(vars_to_marginalize, inplace=False)
                    else: final_factor = filtered_factor
                    print("  Normalizing final factor...")
                    if np.sum(final_factor.values) > 1e-10: final_factor.normalize(inplace=True)
                    else: print("Warning: Probability sum is zero after applying range conditions."); final_factor.values[:] = 0.0
                else: final_factor = joint_factor
                output_list = []; final_factor_vars = final_factor.variables
                final_state_combinations = [final_factor.state_names[var] for var in final_factor_vars]
                final_probabilities = final_factor.values.flatten(); prob_idx = 0
                for state_tuple in itertools.product(*final_state_combinations):
                    row_dict = {}
                    for i, var_name in enumerate(final_factor_vars):
                        state_value = state_tuple[i]
                        if var_name in dag_instance._bool_cols:
                            if state_value.lower() == 'true': state_value = True
                            elif state_value.lower() == 'false': state_value = False
                        row_dict[var_name] = state_value
                    row_dict['probability'] = final_probabilities[prob_idx]
                    output_list.append(row_dict); prob_idx += 1
                return output_list
            except Exception as e:
                print(f"Error during inference (DAG: {dag_instance}) for {query_repr}: {e}")
                import traceback; traceback.print_exc(); raise

    # --- Instantiate P globally ---
    P = P_Calculator()

    # --- Updated DAG Class ---
    class DAG:
        """
        Represents a Directed Acyclic Graph (DAG) learned from data.
        Uses pgmpy for underlying model structure and inference.
        Assumes discrete data or data that has been appropriately discretized.
        Provides Variable objects that store a reference back to this DAG.
        """
        def __init__(self, nodes, edges, dataframe):
            if not isinstance(dataframe, pd.DataFrame): raise ValueError("dataframe must be a pandas DataFrame.")
            # Store nodes in the order provided
            self.nodes = list(nodes) # Ensure it's a list copy
            if not all(node in dataframe.columns for node in self.nodes):
                missing_nodes = [node for node in self.nodes if node not in dataframe.columns]
                raise ValueError(f"Nodes {missing_nodes} not found in DataFrame columns.")
            self.edges = edges
            self.dataframe = dataframe
            self._variables = {} # Cache for Variable objects
            print("Initializing Discrete Bayesian Network model...")
            self.model = DiscreteBayesianNetwork(ebunch=edges)
            self.model.add_nodes_from(self.nodes) # Add nodes in defined order
            print(f"Fitting model to data using {len(dataframe)} samples...")
            df_copy = dataframe.copy()
            self._bool_cols = df_copy.select_dtypes(include=['bool']).columns.tolist()
            for col in self._bool_cols: df_copy[col] = df_copy[col].astype(str)
            self._state_metadata = {}
            if 'age' in nodes: self._state_metadata['age'] = {'type': 'numerical_bin'}
            self.model.fit(df_copy, estimator=MaximumLikelihoodEstimator)
            print("Checking model validity...")
            try:
                if not self.model.check_model(): print("Warning: Model check reported issues.")
                else: print("Model check passed.")
            except Exception as e: print(f"Warning: Model check failed with an error: {e}.")
            print("Initializing inference engine...")
            self.inference = VariableElimination(self.model)
            print("DAG initialization complete.")

        # --- Updated get_variables to return a list preserving original node order ---
        def get_variables(self) -> List[Variable]:
            """
            Returns Variable objects for each node in the DAG, preserving the
            order specified during DAG initialization.

            Returns:
                List[Variable]: A list of Variable objects in the order defined by the 'nodes'
                                argument during DAG initialization.
            """
            # Ensure variables are created if they don't exist, respecting self.nodes order
            if len(self._variables) != len(self.nodes):
                 temp_vars = {}
                 for name in self.nodes:
                     if name not in self._variables:
                         # Create new Variable only if not already cached
                         temp_vars[name] = Variable(name, self)
                     else:
                         # Use existing cached variable
                         temp_vars[name] = self._variables[name]
                 self._variables = temp_vars # Update cache preserving order

            # Return variables in the order specified by self.nodes
            return [self._variables[name] for name in self.nodes]

        def _prepare_evidence(self, evidence_dict):
            # (Remains the same)
            prepared_evidence = {}
            for var, value in evidence_dict.items():
                prepared_evidence[var] = str(value) if var in self._bool_cols else value
            return prepared_evidence

        def P(self, target_variable, evidence=None) -> DiscreteFactor:
            # (Remains the same)
            target_name = target_variable.name if isinstance(target_variable, Variable) else target_variable
            if target_name not in self.nodes: raise ValueError(f"Unknown target variable: {target_name}")
            prepared_evidence = None
            if evidence:
                for var_name in evidence.keys():
                    if var_name not in self.nodes: raise ValueError(f"Unknown evidence variable: {var_name}")
                prepared_evidence = self._prepare_evidence(evidence)
                print(f"Calculating P({target_name} | {prepared_evidence}) using DAG.P method")
            else:
                print(f"Calculating P({target_name}) using DAG.P method")
            try:
                query_result = self.inference.query(variables=[target_name], evidence=prepared_evidence, show_progress=False)
                return query_result
            except Exception as e:
                print(f"Error during inference via DAG.P for P({target_name} | {prepared_evidence}): {e}")
                raise

    return (
        Any,
        Condition,
        DAG,
        Dict,
        DiscreteBayesianNetwork,
        DiscreteFactor,
        List,
        MaximumLikelihoodEstimator,
        P,
        P_Calculator,
        QueryExpression,
        Tuple,
        Union,
        Variable,
        VariableCombination,
        VariableElimination,
        itertools,
        np,
        pd,
        re,
    )


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell(column=1)
def _(pd):
    df_smoking = (pd.read_csv("https://calmcode.io/static/data/smoking.csv")
                  .assign(age=lambda d: (d["age"] / 10).round() * 10))
    return (df_smoking,)


@app.cell
def _(mo):
    mo.md(
        """
        This is what a domain specific language might look like. 

        ```python
        # Define the DAG for the PGM
        dag = DAG(nodes, edges, dataframe)

        # Get variables out
        outcome, smoker, age = dag.get_variables()

        # Use variables to construct a probablistic query
        P(outcome | (smoker == "Yes") & (age > 40))
        ```

        But why stop there?
        """
    )
    return


@app.cell(hide_code=True)
def _(df_smoking, mo):
    from wigglystuff import EdgeDraw

    edge_draw = mo.ui.anywidget(EdgeDraw(list(df_smoking.columns)))
    edge_draw
    return EdgeDraw, edge_draw


@app.cell(hide_code=True)
def _(age_range, alive_no_smoke, alive_smoke, pd):
    pd.DataFrame({
        "age": age_range, 
        "smoke": alive_smoke, 
        "no-smoke": alive_no_smoke
    })
    return


@app.cell
def _(P, age, alt, outcome, pd, smoker):
    age_range = range(10, 70, 10)

    alive_smoke = [
        P(outcome | (smoker == "Yes") & (age > a))[0]["probability"]
        for a in age_range
    ]
    alive_no_smoke = [
        P(outcome | (smoker == "No") & (age > a))[0]["probability"]
        for a in age_range
    ]

    pltr = pd.DataFrame({
        "age": age_range, 
        "smoke": alive_smoke, 
        "no-smoke": alive_no_smoke
    }).melt("age")

    alt.Chart(pltr).mark_line().encode(x="age", y="value", color="variable")
    return age_range, alive_no_smoke, alive_smoke, pltr


@app.cell
def _():
    import altair as alt
    return (alt,)


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(edge_draw):
    [(_['source'], _['target']) for _ in edge_draw.value["links"]]
    return


@app.cell
def _(DAG, df_smoking, edge_draw):
    dag = DAG(
        nodes=edge_draw.value["names"], 
        edges=[(_['source'], _['target']) for _ in edge_draw.value["links"]],
        dataframe=df_smoking
    )
    return (dag,)


@app.cell
def _(dag):
    dag.get_variables()
    return


@app.cell
def _(dag):
    outcome, smoker, age = dag.get_variables()
    return age, outcome, smoker


@app.cell
def _():
    from dicekit import Dice
    return (Dice,)


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(r""" """)
    return


if __name__ == "__main__":
    app.run()
