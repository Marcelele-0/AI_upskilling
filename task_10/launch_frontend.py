"""
Launch script for the RAG System Streamlit Frontend
"""

import subprocess
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def install_requirements():
    """Install required packages for Streamlit frontend"""
    print("📦 Installing Streamlit requirements...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "streamlit_requirements.txt"
        ])
        print("✅ Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False
    return True

def launch_streamlit():
    """Launch the Streamlit application"""
    print("🚀 Launching Streamlit application...")
    
    # Get configuration from environment variables
    port = os.getenv("STREAMLIT_PORT", "8501")
    host = os.getenv("STREAMLIT_HOST", "localhost")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", port,
            "--server.address", host
        ])
    except KeyboardInterrupt:
        print("\n👋 Streamlit application stopped.")
    except Exception as e:
        print(f"❌ Error launching Streamlit: {e}")

def main():
    """Main function"""
    print("🤖 RAG System Frontend Launcher")
    print("=" * 40)
    
    # Check if we're in the correct directory
    if not Path("streamlit_app.py").exists():
        print("❌ streamlit_app.py not found. Please run this script from the project directory.")
        return
    
    # Install requirements if needed
    if not Path("streamlit_requirements.txt").exists():
        print("❌ streamlit_requirements.txt not found.")
        return
        
    # Ask user if they want to install requirements
    install = input("Do you want to install/update requirements? (y/n): ").lower()
    if install in ['y', 'yes']:
        if not install_requirements():
            return
    
    print("\n" + "=" * 40)
    print("🌐 Starting Streamlit Frontend...")
    print("📍 URL: http://localhost:8501")
    print("🛑 Press Ctrl+C to stop")
    print("=" * 40 + "\n")
    
    # Launch Streamlit
    launch_streamlit()

if __name__ == "__main__":
    main()
