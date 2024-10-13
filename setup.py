from setuptools import setup, find_packages
import shutil
import subprocess
import os.path
import sys
import platform

def build_meson():
    # Set the environment variables in Windows for the meson/Fortran compiler to use
    if platform.system() == "Windows":
        if not "FC" in os.environ:
            os.environ["FC"] = "gfortran"
        if not "CC" in os.environ:
            os.environ["CC"] = "gcc"

    # Set the extra compile arguments for the meson build in future 
    # for cross-platform compilation when generating wheels on github runners
    extra_compile_args = []
            
    meson = shutil.which('meson')
    builddir = 'meson_builddir'
    if meson is None:
        raise RuntimeError('meson not found in PATH')
    
    # Remove the old build directory if it exists
    if os.path.exists(builddir):
        shutil.rmtree(builddir)

    # Set up and compile the project using Meson
    subprocess.run([meson, 'setup', *extra_compile_args, builddir], check=True)
    subprocess.run([meson, 'compile', '-C', builddir], check=True)

    build_path = os.path.join(os.getcwd(), builddir, 'pyslsqp')
    target_path = os.path.join(os.getcwd(), 'pyslsqp')

    for root, dirs, files in os.walk(build_path):
        for file in files:
            if file.endswith((".so", ".lib", ".pyd", ".pdb", ".dylib", ".dll")):
                if ".so.p" in root or ".pyd.p" in root:  # excludes intermediate object files in .p directories
                    continue
                from_path = os.path.join(root, file)
                to_path = os.path.join(target_path, file)
                print(f"Copying {from_path} to {to_path}")
                shutil.copy(from_path, to_path)

# ==================================================================================================
# The following forces the generated wheels to be platform specific
# This is necessary because the shared object file is platform specific
# and the wheel would not be installable on other platforms
# if it is marked as a universal wheel (purelib) as setuptools would do by default
# since it is unaware that the package_data is compiled 
# (since it is not compiled as a part of the setup process)

# from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

# class bdist_wheel(_bdist_wheel):
#     def finalize_options(self):
#         # _bdist_wheel.finalize_options(self)
#         super().finalize_options()
#         self.root_is_pure = False
# The `bdist_wheel` class we've defined inherits from `wheel.bdist_wheel.bdist_wheel`, which is the default bdist_wheel command provided by setuptools. 
# In our `bdist_wheel` class, we're overriding the `finalize_options` method to set `self.root_is_pure` to `False`. 
# This tells setuptools that our package is not pure Python, i.e., it contains extensions such as C or Fortran extensions, 
# and therefore the wheel it creates should be platform-specific rather than universal.

from setuptools.dist import Distribution
class BinaryDistribution(Distribution):
    """Distribution which always forces a binary package with platform name"""
    def has_ext_modules(foo):
        return True
# ==================================================================================================

if __name__ == "__main__":
    
    # `build_meson()`` function is only called when the `setup.py`` script is not invoked with the sdist or egg_info command. 
    # This prevents the Fortran compilation using Meson from happening when generating the source distribution.
    # But it will still be executed when installing the package from the source distribution or building the wheel distribution.
    if not ('sdist' in sys.argv or 'egg_info' in sys.argv):
        # Workaround for build_meson() failing on Windows (seems like .pyd is corrupted and no .dll generated)
        if platform.system() == "Windows":
            if not "FC" in os.environ: # Set the environment variables in Windows for the meson/Fortran compiler to use
                os.environ["FC"] = "gfortran"
            if not "CC" in os.environ:
                os.environ["CC"] = "gcc"

            original_dir = os.getcwd()
            os.chdir('pyslsqp/slsqp')
            # sys.executable ensures that the subprocess call uses the same Python interpreter and environment and numpy.f2py is available
            subprocess.run([sys.executable, '-m', 'numpy.f2py', '-c', 'slsqp.pyf', 'slsqp_optmz.f', '--backend', 'meson', '--build-dir', 'meson_builddir'], check=True)
            os.chdir(original_dir)

            build_path  = os.path.join(os.getcwd(), 'pyslsqp', 'slsqp')
            target_path = os.path.join(os.getcwd(), 'pyslsqp')

            for root, dirs, files in os.walk(build_path):
                for file in files:
                    if file.endswith((".so", ".lib", ".pyd", ".pdb", ".dylib", ".dll")):
                        if ".so.p" in root or ".pyd.p" in root:  # excludes intermediate object files in .p directories
                            continue
                        from_path = os.path.join(root, file)
                        to_path = os.path.join(target_path, file)
                        print(f"Copying {from_path} to {to_path}")
                        shutil.copy(from_path, to_path)
            
            # Delete the meson_builddir directory after copying the shared object file
            shutil.rmtree('pyslsqp/slsqp/meson_builddir')
        
        else:
            build_meson()

    setup(
        include_package_data=True,
        package_data={'pyslsqp': ["*.so", "*.lib", "*.pyd", "*.pdb", "*.dylib", "*.dll"]}, # this is needed to include the shared object file in the build directory in site-pkgs
        # cmdclass={'bdist_wheel': bdist_wheel},  # The cmdclass argument in the setup function is used to override default commands provided by setuptools.
        #                                         # This overrides the `bdist_wheel` command. BUT THIS IS SPECIFIC TO WHEELS DISTRIBUTION FORMAT ONLY.
        #                                         # The `bdist_wheel` command is used by setuptools to build a wheel distribution of our package. 
        #                                         # By overriding this command, we can customize how the wheel distribution is built.
        distclass=BinaryDistribution,   # The `distclass` argument in the `setup` function is used to specify a custom distribution class. 
                                        # Basically, you override the 'has_ext_modules' function in the Distribution class, and set distclass to point to the overriding class. 
                                        # At that point, setup.py will believe you have a binary distribution, and will create a wheel with the specific version of python, 
                                        # the ABI, and the current architecture. This is not specific to wheels, and any distribution built with setuptools will be affected.
    )