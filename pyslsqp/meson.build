slsqp_source = custom_target('_slsqpmodule.c',
  input : ['slsqp/slsqp.pyf'],
  output : ['_slsqpmodule.c'],
  command : [py, '-m', 'numpy.f2py', '@INPUT@', '--lower', '--build-dir', 'pyslsqp/']
)

py.extension_module('_slsqp',
  ['slsqp/slsqp_optmz.f', slsqp_source],      # Source files for the extension module
  incdir_f2py / 'fortranobject.c',            # Include directories or additional files
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