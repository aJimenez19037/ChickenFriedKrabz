import sys
import os
import importlib

def test_protobuf_imports():
    # Define paths to protobuf_old and protobuf_new
    protobuf_old_path = os.path.expanduser('~/protobuf_old')
    protobuf_new_path = os.path.expanduser('~/protobuf_new')

    try:
        # Test MAVSDK with protobuf_old
        print("Testing MAVSDK with protobuf_old...")
        sys.path.insert(0, protobuf_old_path)
        import mavsdk
        from google.protobuf import __version__ as protobuf_version_old
        print(f"MAVSDK loaded successfully with protobuf version: {protobuf_version_old}")
        # Use MAVSDK after TensorFlow loads protobuf 3.20.3
        sys.path.remove(protobuf_old_path)

    except Exception as e:
        print(f"Error testing MAVSDK with protobuf_old: {e}")

    try:
        # Test TensorFlow with protobuf_new
        print("\nTesting TensorFlow with protobuf_new...")
        sys.path.insert(0, protobuf_new_path)
        importlib.reload(sys.modules["google.protobuf"])  # Reload protobuf
        import tensorflow as tf
        from google.protobuf import __version__ as protobuf_version_new
        print(f"TensorFlow loaded successfully with protobuf version: {protobuf_version_new}")
        sys.path.remove(protobuf_new_path)

    except Exception as e:
        print(f"Error testing TensorFlow with protobuf_new: {e}")

    try:
        # Reuse MAVSDK after TensorFlow loads protobuf_new
        print("\nReusing MAVSDK to confirm it still works...")
        from mavsdk import System
        print("MAVSDK is still functional with protobuf 3.20.1.")

    except Exception as e:
        print(f"Error reusing MAVSDK after TensorFlow: {e}")

if __name__ == "__main__":
    test_protobuf_imports()
