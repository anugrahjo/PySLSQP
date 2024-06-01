
import re

def extract_python_code_from_md(filepath):
    '''
    Extracts python code snippets as a list from a markdown file.
    Each code snippet is a string in the list.
    '''
    with open(filepath, 'r') as file:
        content = file.read()
    python_code = re.findall(r'```python(.*?)```', content, re.DOTALL)
    return python_code

import json

def extract_python_code_from_nb(filepath):
    '''
    Extracts python code snippets as a list from a Jupyter notebook file.
    Each code snippet is a string in the list.
    '''
    with open(filepath, 'r') as file:
        notebook = json.load(file)
    # if not omitting lines starting with '%'
    # python_code = [cell['source'] for cell in notebook['cells'] if cell['cell_type'] == 'code']
    python_code = []
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            cell_code = ''.join(line for line in cell['source'] if not line.startswith('%'))
            python_code.append(cell_code)
    
    return python_code

def execute_python_code_snippet(code, global_vars=globals()):
    '''
    Executes one Python code snippet in presence of given global variables
    and returns a dictionary containing local variables.
    '''
    local_vars = {}
    exec(code, global_vars, local_vars)
    # For executing a list of snippets
    # for snippet in code:
    #     exec(snippet, globals(), local_vars)
    return local_vars

def check_timing(results, visualize=False):
    '''
    Check if timing results are positive and make sense.
    '''
    assert results['fev_time'] > 0
    assert results['gev_time'] > 0
    assert results['optimizer_time'] > 0
    assert results['processing_time'] > 0
    if not visualize:
        assert results['visualization_time'] == 0.0
    else:
        assert results['visualization_time'] > 0
    
    assert results['total_time'] > 0
    assert results['total_time'] > results['optimizer_time']
    assert results['total_time'] > results['fev_time']
    assert results['total_time'] > results['gev_time']
    assert results['total_time'] > results['processing_time']
    assert results['total_time'] > results['visualization_time']
    assert results['total_time'] == results['optimizer_time'] + results['fev_time'] + results['gev_time'] + results['processing_time'] + results['visualization_time']