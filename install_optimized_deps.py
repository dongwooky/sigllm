#!/usr/bin/env python3
"""
Install additional dependencies for optimized Together AI pipeline
"""

import subprocess
import sys

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Successfully installed {package}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False
    return True

def main():
    """Install required dependencies for optimized pipeline"""
    print("=== Installing Optimized Pipeline Dependencies ===")
    
    # Required packages for optimization
    packages = [
        "aiohttp",           # For async HTTP requests
        "tqdm",              # For progress bars (if not already installed)
        "asyncio",           # Built-in, but ensure it's available
    ]
    
    failed_packages = []
    
    for package in packages:
        if package == "asyncio":
            # asyncio is built-in, just check if it's available
            try:
                import asyncio
                print(f"✅ {package} is available (built-in)")
            except ImportError:
                print(f"❌ {package} not available")
                failed_packages.append(package)
        else:
            if not install_package(package):
                failed_packages.append(package)
    
    if failed_packages:
        print(f"\n❌ Failed to install: {failed_packages}")
        print("Please install manually using:")
        for pkg in failed_packages:
            print(f"  pip install {pkg}")
        return False
    else:
        print("\n✅ All dependencies installed successfully!")
        print("You can now run the optimized pipeline:")
        print("  python mistral-detector-pipeline_together_optimized.py")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
