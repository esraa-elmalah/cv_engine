#!/usr/bin/env python3
"""
CV Engine - Setup Script
Helps set up the project environment
"""

import os
import sys
import subprocess
import venv
from pathlib import Path

def print_banner():
    """Print the setup banner."""
    print("=" * 60)
    print("🔧 CV Engine - Setup Script")
    print("=" * 60)
    print()

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    
    print(f"✅ Python {sys.version.split()[0]} is compatible")
    return True

def install_dependencies():
    """Install project dependencies."""
    print("📦 Installing dependencies...")
    
    try:
        subprocess.run([
            "python3", "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def setup_backend():
    """Setup backend environment."""
    print("🔧 Setting up backend...")
    
    backend_dir = Path("backend")
    if not backend_dir.exists():
        print("❌ Backend directory not found")
        return False
    
    # Install backend dependencies using python3
    print("📦 Installing backend dependencies...")
    try:
        subprocess.run([
            "python3", "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True)
        print("✅ Backend dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install backend dependencies: {e}")
        return False

def setup_frontend():
    """Setup frontend environment."""
    print("🔧 Setting up frontend...")
    
    frontend_dir = Path("frontend")
    if not frontend_dir.exists():
        print("❌ Frontend directory not found")
        return False
    
    # Create .env file if it doesn't exist
    env_file = frontend_dir / ".env"
    if not env_file.exists():
        config_example = frontend_dir / "config.example"
        if config_example.exists():
            print("📝 Creating frontend .env file...")
            with open(config_example, 'r') as f:
                content = f.read()
            with open(env_file, 'w') as f:
                f.write(content)
            print("✅ Frontend .env file created")
    
    return True

def main():
    """Main setup function."""
    print_banner()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    print()
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Setup failed at dependency installation")
        sys.exit(1)
    
    print()
    
    # Setup backend
    if not setup_backend():
        print("❌ Setup failed at backend setup")
        sys.exit(1)
    
    print()
    
    # Setup frontend
    if not setup_frontend():
        print("❌ Setup failed at frontend setup")
        sys.exit(1)
    
    print()
    print("🎉 Setup completed successfully!")
    print("=" * 60)
    print("Next steps:")
    print("1. Configure your environment variables:")
    print("   - Copy backend/.env.example to backend/.env")
    print("   - Add your OpenAI API key")
    print()
    print("2. Start the system:")
    print("   python start.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
