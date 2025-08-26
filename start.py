#!/usr/bin/env python3
"""
CV Engine Startup Script
Launches both backend and frontend services.
"""

import os
import sys
import time
import subprocess
from pathlib import Path

def print_banner():
    """Print startup banner."""
    print("=" * 60)
    print("🤖 CV Engine - AI-Powered CV Generation & Analysis")
    print("=" * 60)

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import fastapi
        import uvicorn
        import openai
        import gradio
        import requests
        import dotenv
        import pydantic
        import pydantic_settings
        import weasyprint
        import faiss
        import langchain_community
        import langchain_openai
        import pypdf
        import tenacity
        import pytest
        import pytest_asyncio
        import pytest_cov
        import httpx
        import agents
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run: python3 setup.py")
        return False

def start_backend():
    """Start the backend server."""
    print("🚀 Starting Backend Server...")
    backend_dir = Path("backend")
    
    if not backend_dir.exists():
        print("❌ Backend directory not found")
        return False
    
    try:
        # Use virtual environment
        venv_python = backend_dir / "venv" / "bin" / "python"
        if venv_python.exists():
            print(f"✅ Using virtual environment: {venv_python}")
            process = subprocess.Popen([
                str(venv_python.resolve()), "-m", "uvicorn", 
                "app.main:app", 
                "--reload", 
                "--host", "0.0.0.0", 
                "--port", "8000"
            ], cwd=backend_dir)
        else:
            print("⚠️ Virtual environment not found, using system python")
            process = subprocess.Popen([
                "python3", "-m", "uvicorn", 
                "app.main:app", 
                "--reload", 
                "--host", "0.0.0.0", 
                "--port", "8000"
            ], cwd=backend_dir)
        
        print("✅ Backend started on http://localhost:8000")
        print("📚 API Documentation: http://localhost:8000/docs")
        return process
        
    except Exception as e:
        print(f"❌ Failed to start backend: {e}")
        return False

def start_frontend():
    """Start the frontend server."""
    print("🎨 Starting Frontend Server...")
    frontend_dir = Path("frontend")
    
    if not frontend_dir.exists():
        print("❌ Frontend directory not found")
        return False
    
    try:
        # Use virtual environment for frontend too
        venv_python = Path("backend/venv/bin/python")
        if venv_python.exists():
            print(f"✅ Using virtual environment for frontend: {venv_python}")
            process = subprocess.Popen([
                str(venv_python.resolve()), "start.py"
            ], cwd=frontend_dir)
        else:
            print("⚠️ Virtual environment not found for frontend")
            process = subprocess.Popen([
                "python3", "start.py"
            ], cwd=frontend_dir)
        
        print("✅ Frontend started on http://localhost:7860")
        return process
        
    except Exception as e:
        print(f"❌ Failed to start frontend: {e}")
        return False

def main():
    """Main startup function."""
    print_banner()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    print("🔍 Checking project structure...")
    
    # Check if we're in the right directory
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found. Please run this script from the project root.")
        sys.exit(1)
    
    print("✅ Project structure looks good")
    print()
    
    # Store original directory
    original_dir = os.getcwd()
    processes = []
    
    try:
        # Start backend
        backend_process = start_backend()
        if backend_process:
            processes.append(("Backend", backend_process))
            time.sleep(2)  # Give backend time to start
        
        # Start frontend
        os.chdir(original_dir)  # Return to original directory
        frontend_process = start_frontend()
        if frontend_process:
            processes.append(("Frontend", frontend_process))
        
        if not processes:
            print("❌ Failed to start any services")
            sys.exit(1)
        
        print()
        print("🎉 CV Engine is running!")
        print("=" * 60)
        print("📱 Frontend: http://localhost:7860")
        print("🔧 Backend:  http://localhost:8000")
        print("📚 API Docs: http://localhost:8000/docs")
        print("=" * 60)
        print("🛑 Press Ctrl+C to stop all services")
        print()
        
        # Wait for user to stop
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping services...")
            
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        sys.exit(1)
    
    finally:
        # Cleanup processes
        for name, process in processes:
            try:
                print(f"🛑 Stopping {name}...")
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(f"⚠️ Force killing {name}...")
                process.kill()
            except Exception as e:
                print(f"⚠️ Error stopping {name}: {e}")
        
        print("👋 CV Engine stopped")

if __name__ == "__main__":
    main()
