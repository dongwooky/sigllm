#!/usr/bin/env python3
"""
Wrapper script to run Together AI pipeline with dependency conflict workaround
"""

import os
import sys

# Ignore package version conflicts
os.environ['PYTHONWARNINGS'] = 'ignore'

# Monkey patch pkg_resources to ignore version conflicts
try:
    import pkg_resources
    original_require = pkg_resources.WorkingSet.require
    
    def patched_require(self, *requirements, **kwargs):
        try:
            return original_require(self, *requirements, **kwargs)
        except pkg_resources.VersionConflict:
            # Ignore version conflicts and continue
            pass
    
    pkg_resources.WorkingSet.require = patched_require
    print("✅ Applied version conflict workaround")
except ImportError:
    pass

# Now import and run the pipeline
try:
    print("🚀 Starting Together AI pipeline...")
    exec(open('tutorials/pipelines/mistral-detector-pipeline_together.py').read())
except Exception as e:
    print(f"❌ Error: {e}")
    print("\n💡 Make sure to set TOGETHER_API_KEY environment variable:")
    print("export TOGETHER_API_KEY='your-api-key-here'")
