import subprocess
import sys

def install_if_missing(package):
    try:
        __import__(package)
        print(f"✅ {package} is already installed.")
    except ImportError:
        print(f"📦 Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# List packages required
required_packages = {
    "selenium": "selenium",
    "webdriver_manager": "webdriver-manager"
}

# Verify and install each package
for import_name, pip_name in required_packages.items():
    install_if_missing(import_name)