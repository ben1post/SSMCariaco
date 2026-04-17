"""
Runtime hooks for capturing stability analysis results from XSO solvers.

These hooks extract stability metadata from the solver after the simulation
completes and make it available for adding to the output dataset.
"""

import xsimlab as xs
import numpy as np


# Storage for the last stability results
_stability_results = {}


@xs.runtime_hook("finalize", "model", "post")
def capture_stability_metadata(model, context, state):
    """
    Runtime hook that captures stability analysis results after the model finishes.
    
    This hook is called after the 'finalize' stage, when the solver has completed
    its analysis. It extracts stability metadata from the solver if available.
    
    Parameters
    ----------
    model : xsimlab.Model
        The model instance that just finished running
    context : dict
        Runtime context information
    state : dict
        Model state with keys as ('process_name', 'variable_name') tuples
    """
    global _stability_results
    
    # Clear previous results
    _stability_results = {}
    
    try:
        # Access the solver through the model's backend
        # The Backend process stores the core, which contains the solver
        backend_state = state.get(('Core', 'core'))
        
        if backend_state is not None:
            # Check if solver has stability results
            solver = backend_state.solver
            
            # Check if this is a stability solver with results
            if hasattr(solver, 'stability_results'):
                _stability_results = solver.stability_results.copy()
                print(f"[HOOK] Captured stability analysis results")
                
            # Alternative: check for results stored in the core itself
            elif hasattr(backend_state, 'stability_metadata'):
                _stability_results = backend_state.stability_metadata.copy()
                print(f"[HOOK] Captured stability metadata from core")
                
    except (AttributeError, KeyError) as e:
        # Solver doesn't have stability results or structure is different
        pass


def get_stability_results():
    """
    Retrieve the last captured stability analysis results.
    
    Returns
    -------
    dict
        Dictionary containing stability analysis results, or empty dict if none available.
    """
    return _stability_results.copy()


def clear_stability_results():
    """Clear the stored stability results."""
    global _stability_results
    _stability_results = {}


# Alternative: Class-based approach for more control
class StabilityAnalysisHook(xs.RuntimeHook):
    """
    Runtime hook class for capturing stability analysis results.
    
    This class-based approach allows for more control and state management
    compared to the function-based approach.
    
    Usage
    -----
    >>> hook = StabilityAnalysisHook()
    >>> with model:
    ...     output = model_setup.xsimlab.run(hooks=[hook])
    >>> 
    >>> # Retrieve results
    >>> stability_data = hook.get_results()
    >>> if stability_data:
    ...     output.attrs['stability'] = stability_data
    """
    
    def __init__(self):
        super().__init__()
        self.stability_results = {}
        self._solver_found = False
    
    @xs.runtime_hook("initialize", "model", "post")
    def check_solver_type(self, model, context, state):
        """Check if the solver supports stability analysis at initialization."""
        try:
            backend = state.get(('Core', 'core'))
            if backend and hasattr(backend.solver, '__class__'):
                solver_name = backend.solver.__class__.__name__
                if 'Stability' in solver_name or 'Bifurcation' in solver_name:
                    self._solver_found = True
                    print(f"[HOOK] Detected {solver_name} - will capture stability results")
        except:
            pass
    
    @xs.runtime_hook("finalize", "model", "post")  
    def capture_results(self, model, context, state):
        """Capture stability results after model finalization."""
        if not self._solver_found:
            return
            
        try:
            backend = state.get(('Core', 'core'))
            
            if backend:
                solver = backend.solver
                
                # Try different storage locations
                if hasattr(solver, 'stability_results'):
                    self.stability_results = solver.stability_results.copy()
                elif hasattr(backend, 'stability_metadata'):
                    self.stability_results = backend.stability_metadata.copy()
                
                if self.stability_results:
                    self._print_summary()
                    
        except Exception as e:
            print(f"[HOOK WARNING] Could not capture stability results: {e}")
    
    def _print_summary(self):
        """Print a summary of captured stability results."""
        if 'stability' in self.stability_results:
            print(f"[HOOK] System stability: {self.stability_results['stability'].upper()}")
        if 'max_eigenvalue_real' in self.stability_results:
            print(f"[HOOK] Max eigenvalue real part: {self.stability_results['max_eigenvalue_real']:.4e}")
    
    def get_results(self):
        """Get the captured stability analysis results."""
        return self.stability_results.copy()
    
    def clear(self):
        """Clear stored results."""
        self.stability_results = {}
        self._solver_found = False


# Convenience function to create and use the hook
def run_with_stability_analysis(model, model_setup, **run_kwargs):
    """
    Convenience function to run a model and capture stability analysis results.
    
    Parameters
    ----------
    model : xsimlab.Model
        The model to run
    model_setup : xarray.Dataset
        The setup dataset for the model
    **run_kwargs
        Additional keyword arguments to pass to xsimlab.run()
    
    Returns
    -------
    xarray.Dataset
        Output dataset with stability analysis in attributes if available
    
    Example
    -------
    >>> output = run_with_stability_analysis(model, model_setup)
    >>> print(output.attrs.get('stability_analysis', 'No stability analysis performed'))
    """
    hook = StabilityAnalysisHook()
    
    # Add hook to any existing hooks
    hooks = run_kwargs.get('hooks', [])
    hooks = list(hooks) + [hook]
    run_kwargs['hooks'] = hooks
    
    with model:
        output = model_setup.xsimlab.run(**run_kwargs)
    
    # Add stability results to output attributes
    stability_data = hook.get_results()
    if stability_data:
        output.attrs['stability_analysis'] = stability_data
    
    return output