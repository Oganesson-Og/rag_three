import sys
import pkg_resources

def check_dependencies():
    required_packages = {
        'numpy': '1.21.0',
        'pandas': '1.3.0',
        'pydantic': '2.0.0',
        'pytest': '7.0.0',
        'PyYAML': '6.0.0',
        'sentence-transformers': '2.2.0',
        'torch': '2.0.0',
        'transformers': '4.30.0',
        'whisper-openai': '1.0.0',
        'librosa': '0.10.0',
        'soundfile': '0.12.0',
        'python-dotenv': '0.19.0',
        'tqdm': '4.65.0'
    }
    
    missing_packages = []
    version_mismatch = []
    
    for package, min_version in required_packages.items():
        try:
            installed = pkg_resources.get_distribution(package)
            if pkg_resources.parse_version(installed.version) < pkg_resources.parse_version(min_version):
                version_mismatch.append(f"{package} version {installed.version} is lower than required version {min_version}")
        except pkg_resources.DistributionNotFound:
            missing_packages.append(package)
    
    return missing_packages, version_mismatch

def check_system_dependencies():
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
    except Exception as e:
        print(f"PyTorch check failed: {str(e)}")
        
    try:
        import whisper
        print("Whisper import successful")
    except Exception as e:
        print(f"Whisper check failed: {str(e)}")
        
    try:
        import soundfile as sf
        print("SoundFile import successful")
    except Exception as e:
        print(f"SoundFile check failed: {str(e)}")

if __name__ == "__main__":
    print("Python version:", sys.version)
    print("\nChecking package dependencies...")
    missing, version_issues = check_dependencies()
    
    if missing:
        print("\nMissing packages:")
        for pkg in missing:
            print(f"- {pkg}")
    
    if version_issues:
        print("\nVersion issues:")
        for issue in version_issues:
            print(f"- {issue}")
            
    if not missing and not version_issues:
        print("All required packages are installed with correct versions!")
    
    print("\nChecking system dependencies...")
    check_system_dependencies()