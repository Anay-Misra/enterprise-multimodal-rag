#!/usr/bin/env python3
"""
Diagnostic script to identify the exact import issue
"""

import os
import sys
from pathlib import Path

def check_file_exists(filename):
    """Check if file exists in current directory"""
    current_dir = Path(".")
    file_path = current_dir / filename
    exists = file_path.exists()
    print(f"{'✅' if exists else '❌'} {filename}: {'Exists' if exists else 'Missing'}")
    if exists:
        size = file_path.stat().st_size
        print(f"    Size: {size} bytes")
    return exists

def test_import_step_by_step():
    """Test imports step by step to identify the exact issue"""
    print("🔍 Step-by-step Import Test")
    print("-" * 30)
    
    # Test basic agentic_rag import
    try:
        import agentic_rag
        print("✅ Basic agentic_rag import successful")
    except Exception as e:
        print(f"❌ Basic agentic_rag import failed: {e}")
        return False
    
    # Test individual imports from agentic_rag
    imports_to_test = [
        "test_graph_rag_setup",
        "GRAPH_RAG_AVAILABLE", 
        "NEO4J_AVAILABLE",
        "MULTIMODAL_AVAILABLE"
    ]
    
    for import_name in imports_to_test:
        try:
            value = getattr(agentic_rag, import_name)
            print(f"✅ {import_name}: {value}")
        except AttributeError:
            print(f"❌ {import_name}: Not found in agentic_rag module")
        except Exception as e:
            print(f"❌ {import_name}: Error accessing - {e}")
    
    # Test the function call
    try:
        result = agentic_rag.test_graph_rag_setup()
        print(f"✅ test_graph_rag_setup() result: {result}")
        return True
    except Exception as e:
        print(f"❌ test_graph_rag_setup() failed: {e}")
        return False

def check_multimodal_files():
    """Check if multimodal files exist and can be imported"""
    print("\n🎭 Multimodal Files Check")
    print("-" * 25)
    
    multimodal_files = [
        "multimodal_processors.py",
        "multimodal_readers.py"
    ]
    
    all_exist = True
    for filename in multimodal_files:
        exists = check_file_exists(filename)
        if not exists:
            all_exist = False
    
    if all_exist:
        # Test imports
        try:
            import multimodal_processors
            print("✅ multimodal_processors import successful")
        except Exception as e:
            print(f"❌ multimodal_processors import failed: {e}")
        
        try:
            import multimodal_readers
            print("✅ multimodal_readers import successful")
        except Exception as e:
            print(f"❌ multimodal_readers import failed: {e}")
    
    return all_exist

def show_agentic_rag_content():
    """Show what's actually in the agentic_rag module"""
    print("\n📄 agentic_rag.py Content Analysis")
    print("-" * 35)
    
    try:
        import agentic_rag
        
        # Show all attributes
        attributes = [attr for attr in dir(agentic_rag) if not attr.startswith('_')]
        print(f"Available attributes: {attributes}")
        
        # Check specific ones we need
        needed_attrs = ['MULTIMODAL_AVAILABLE', 'test_graph_rag_setup']
        for attr in needed_attrs:
            if hasattr(agentic_rag, attr):
                value = getattr(agentic_rag, attr)
                print(f"✅ {attr}: {value}")
            else:
                print(f"❌ {attr}: Missing")
        
        return True
    except Exception as e:
        print(f"❌ Error analyzing agentic_rag: {e}")
        return False

def check_python_path():
    """Check Python path and working directory"""
    print("\n🐍 Python Environment")
    print("-" * 20)
    
    print(f"Working directory: {os.getcwd()}")
    print(f"Python path: {sys.path[:3]}...")  # Show first 3 entries
    
    # Check if current directory is in Python path
    current_dir = str(Path(".").resolve())
    if current_dir in sys.path:
        print("✅ Current directory is in Python path")
    else:
        print("❌ Current directory NOT in Python path")
        print("   This might cause import issues")

def main():
    """Run all diagnostics"""
    print("🔧 Multimodal Import Diagnostics")
    print("=" * 40)
    
    # Check Python environment
    check_python_path()
    
    # Check if files exist
    print("\n📁 File Existence Check")
    print("-" * 25)
    
    required_files = [
        "agentic_rag.py",
        "multimodal_processors.py", 
        "multimodal_readers.py",
        "app.py"
    ]
    
    files_exist = []
    for filename in required_files:
        exists = check_file_exists(filename)
        files_exist.append(exists)
    
    if not all(files_exist):
        print("\n❌ Some required files are missing!")
        print("Please make sure you have created all the multimodal files in your project directory.")
        return
    
    # Check multimodal files
    check_multimodal_files()
    
    # Check agentic_rag content
    show_agentic_rag_content()
    
    # Test step-by-step import
    test_import_step_by_step()
    
    print("\n" + "=" * 40)
    print("🎯 Diagnosis Complete")
    print("=" * 40)
    
    # Give specific recommendations
    print("\n💡 Recommendations:")
    
    try:
        import agentic_rag
        if not hasattr(agentic_rag, 'MULTIMODAL_AVAILABLE'):
            print("1. ❌ MULTIMODAL_AVAILABLE is missing from agentic_rag.py")
            print("   → Replace agentic_rag.py with the updated version")
        else:
            print("1. ✅ MULTIMODAL_AVAILABLE found in agentic_rag.py")
    except:
        print("1. ❌ Cannot import agentic_rag.py at all")
        print("   → Check for syntax errors in agentic_rag.py")
    
    try:
        import multimodal_processors
        import multimodal_readers
        print("2. ✅ Multimodal files can be imported")
    except Exception as e:
        print(f"2. ❌ Multimodal files have import issues: {e}")
        print("   → Check multimodal_processors.py and multimodal_readers.py for errors")

if __name__ == "__main__":
    main()