def windows_dll_setup_for_pysr():
    import os
    if os.name == 'nt':
        import ctypes
        import glob

        # Path to the bin directory of your Julia installation
        julia_bin_path = r"C:\Users\titus\.julia\juliaup\julia-1.10.4+0.x64.w64.mingw32\bin"  # (which is the same as the julia.exe)

        # Add the bin directory to PATH
        os.environ["PATH"] += ";" + julia_bin_path

        # Load each DLL file in the bin directory
        for dll_path in glob.glob(os.path.join(julia_bin_path, "*.dll")):
            try:
                ctypes.CDLL(dll_path)
                print(f"Loaded {dll_path} successfully.")
            except OSError as e:
                print(f"Could not load {dll_path}: {e}")
