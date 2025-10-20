#!/usr/bin/env python3
"""
Verification script for BulkFormer installation.
Run this to check if all dependencies are properly installed.
"""

import sys
from typing import List, Tuple


def check_import(module_name: str, package_name: str = None) -> Tuple[bool, str]:
    """
    Try to import a module and return status.
    
    Args:
        module_name: Name of the module to import
        package_name: Display name (if different from module_name)
    
    Returns:
        Tuple of (success, version_or_error)
    """
    if package_name is None:
        package_name = module_name
    
    try:
        module = __import__(module_name)
        version = getattr(module, '__version__', 'installed')
        return True, version
    except ImportError as e:
        return False, str(e)


def main() -> int:
    """Run verification checks."""
    print("BulkFormer Installation Verification")
    print("=" * 60)
    
    # Core dependencies
    checks: List[Tuple[str, str]] = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('torchaudio', 'TorchAudio'),
        ('torch_geometric', 'PyTorch Geometric'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('scipy', 'SciPy'),
        ('sklearn', 'Scikit-learn'),
        ('matplotlib', 'Matplotlib'),
        ('scanpy', 'Scanpy'),
        ('anndata', 'AnnData'),
        ('performer_pytorch', 'Performer PyTorch'),
        ('einops', 'Einops'),
        ('transformers', 'Transformers'),
    ]
    
    # Project modules
    project_checks: List[Tuple[str, str]] = [
        ('bulkformer', 'BulkFormer Package'),
        ('bulkformer.models.model', 'BulkFormer Module'),
        ('bulkformer.models.encoder', 'Encoder Block'),
        ('bulkformer.models.rope', 'RoPE Embedding'),
        ('bulkformer.config', 'Model Config'),
    ]
    
    all_passed = True
    
    print("\nCore Dependencies:")
    print("-" * 60)
    for module, display_name in checks:
        success, info = check_import(module, display_name)
        status = "✓" if success else "✗"
        print(f"{status} {display_name:<25} {info}")
        if not success:
            all_passed = False
    
    print("\nProject Modules:")
    print("-" * 60)
    for module, display_name in project_checks:
        success, info = check_import(module, display_name)
        status = "✓" if success else "✗"
        print(f"{status} {display_name:<25} {'OK' if success else info}")
        if not success:
            all_passed = False
    
    # Check PyTorch CUDA
    print("\nPyTorch CUDA Status:")
    print("-" * 60)
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"{'✓' if cuda_available else '✗'} CUDA Available: {cuda_available}")
        if cuda_available:
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  Number of GPUs: {torch.cuda.device_count()}")
            if torch.cuda.device_count() > 0:
                print(f"  GPU 0: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"✗ Error checking CUDA: {e}")
        all_passed = False
    
    # Optional dependencies
    print("\nOptional Dependencies (for better performance):")
    print("-" * 60)
    optional_checks = [
        ('torch_cluster', 'torch-cluster'),
        ('torch_scatter', 'torch-scatter'),
        ('torch_sparse', 'torch-sparse'),
        ('torch_spline_conv', 'torch-spline-conv'),
    ]
    
    for module, display_name in optional_checks:
        success, info = check_import(module, display_name)
        status = "✓" if success else "○"
        message = info if success else "not installed (optional)"
        print(f"{status} {display_name:<25} {message}")
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All required dependencies are installed correctly!")
        print("\nYou can now run:")
        print("  uv run jupyter notebook notebooks/bulkformer_extract_feature.ipynb")
        return 0
    else:
        print("✗ Some dependencies are missing or failed to import.")
        print("\nTry running:")
        print("  uv sync")
        return 1


if __name__ == '__main__':
    sys.exit(main())

