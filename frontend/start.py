#!/usr/bin/env python3
"""
CV Engine Frontend Startup Script
"""

import os
import sys
import subprocess
import time
import requests
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import gradio
        import requests
        import dotenv
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run: pip install -r ../requirements.txt")
        return False

def check_backend_health(api_url: str) -> bool:
    """Check if the backend is running and healthy."""
    try:
        response = requests.get(f"{api_url}/api/v1/stats", timeout=5)
        if response.status_code == 200:
            print("✅ Backend is running and healthy")
            return True
        else:
            print(f"⚠️ Backend returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to backend")
        print("Please ensure the backend is running on port 8000")
        return False
    except Exception as e:
        print(f"❌ Backend health check failed: {e}")
        return False

def setup_environment():
    """Setup environment variables."""
    # Check if .env file exists, if not create from example
    env_file = Path(".env")
    if not env_file.exists():
        config_example = Path("config.example")
        if config_example.exists():
            print("📝 Creating .env file from config.example")
            with open(config_example, 'r') as f:
                content = f.read()
            with open(env_file, 'w') as f:
                f.write(content)
        else:
            print("📝 Creating default .env file")
            with open(env_file, 'w') as f:
                f.write("API_BASE_URL=http://127.0.0.1:8000\n")
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    api_url = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
    print(f"🔗 Using API URL: {api_url}")
    return api_url

def main():
    """Main startup function."""
    print("🚀 Starting CV Engine Frontend...")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Setup environment
    api_url = setup_environment()
    
    # Check backend health
    print("\n🔍 Checking backend health...")
    if not check_backend_health(api_url):
        print("\n💡 To start the backend, run:")
        print("   cd ../backend")
        print("   source venv/bin/activate")
        print("   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
        print("\n⚠️ Frontend will start anyway, but backend features won't work")
    
    # Start the frontend
    print("\n🌐 Starting frontend server...")
    print("📱 Frontend will be available at: http://localhost:7860")
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        from ui import launch_ui
        launch_ui()
    except KeyboardInterrupt:
        print("\n👋 Frontend stopped by user")
    except Exception as e:
        print(f"\n❌ Frontend startup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
