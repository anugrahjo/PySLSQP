slsqp_source = custom_target('_slsqpmodule.c',
  input : ['slsqp/slsqp.pyf'],
  output : ['_slsqpmodule.c'],
  command : [py, '-m', 'numpy.f2py', '@INPUT@', '--lower', '--build-dir', 'pyslsqp/']
)

# Optimization flags for the Fortran compiler
# '-ffp-contract=off' (for GCC/Clang) is to avoid floating-point optimizations that can cause problems with macOS-14 (M1 chips) 
# '-ffp-contract=off' prevents aggressive contraction of floating-point operations
fortran_optimization_flags = ['-O3', '-ffp-contract=off']

# Using the lowest optimization level '-O1' also prevents tests from failing with the M1 chip
# fortran_optimization_flags = ['-O1']

py.extension_module('_slsqp',
  ['slsqp/slsqp_optmz.f', slsqp_source],      # Source files for the extension module
  incdir_f2py / 'fortranobject.c',            # Include directories or additional files
  fortran_args: fortran_optimization_flags,   # Fortran compiler arguments
  include_directories: inc_np,                # Include directories for dependencies
  dependencies : py_dep,                      # Dependency objects (like libraries)
  # dependencies : fortranobject_dep,         # Dependency objects (like libraries)
  subdir : 'pyslsqp/',                        # Subdirectory for organizing the source
  # link_language: 'fortran',                 # For meson to generate the linked .dll files for Windows (didn't work)
  install : false                             # Do not install this module when running `meson install`
  # install : true                            # Install this module when running `meson install`
)                                             # Use `install=True` when you want the compiled module, executable, or other target to be part of the final installation. 
                                              # This is common for finished products or components that need to be available system-wide or within a specified environment.
                                              # This ensures that the _slsqp module is copied to the appropriate installation directory when you run the `meson install` command.