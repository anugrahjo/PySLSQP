project('pyslsqp', 'c',
  version : '0.1',
  license: 'BSD-3',
  meson_version: '>=0.64.0',
  default_options : ['warning_level=2'],
)

add_languages('fortran', native: false)

py_mod = import('python')
py = py_mod.find_installation(pure: false)
py_dep = py.dependency()

incdir_numpy = run_command(py,
  ['-c', 'import os; os.chdir(".."); import numpy; print(numpy.get_include())'],
  check : true
).stdout().strip()

# this creates a raw string which is useful for Windows use of '\' for paths
incdir_numpy = '''@0@'''.format(incdir_numpy)

incdir_f2py = run_command(py,
    ['-c', 'import os; os.chdir(".."); import numpy.f2py; print(numpy.f2py.get_include())'],
    check : true
).stdout().strip()

inc_np = include_directories(incdir_numpy, incdir_f2py)

# fortranobject_c = incdir_f2py / 'fortranobject.c'

# # Share this object across multiple modules.
# fortranobject_lib = static_library('_fortranobject',
#   fortranobject_c,
#   dependencies: py_dep,
#   include_directories: inc_np,
# )
# fortranobject_dep = declare_dependency(
#   link_with: fortranobject_lib,
#   include_directories: inc_np,
# )

subdir('pyslsqp')