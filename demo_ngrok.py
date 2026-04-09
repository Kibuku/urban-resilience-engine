"""
Demo script for Urban Resilience Engine using ngrok.
Creates a secure tunnel to expose the Streamlit dashboard for demos.

Usage:
1. Install ngrok: pip install pyngrok
2. Sign up at ngrok.com and get your auth token
3. Set NGROK_AUTH_TOKEN in your .env file
4. Run: python demo_ngrok.py

The dashboard will be available at the ngrok URL for 8 hours.
"""
import os
import sys
import subprocess
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def setup_ngrok():
    """Set up ngrok authentication."""
    from dotenv import load_dotenv
    load_dotenv(override=True)

    auth_token = os.getenv("NGROK_AUTH_TOKEN")
    if not auth_token:
        print(" NGROK_AUTH_TOKEN not found in .env file")
        print("   Please:")
        print("   1. Sign up at https://ngrok.com")
        print("   2. Get your auth token from https://dashboard.ngrok.com/get-started/your-authtoken")
        print("   3. Add it to your .env file: NGROK_AUTH_TOKEN=your_token_here")
        return False

    if auth_token.strip() == "your-ngrok-auth-token":
        print(" NGROK_AUTH_TOKEN is still the placeholder value.")
        print("   Replace it with the real ngrok token from https://dashboard.ngrok.com/get-started/your-authtoken")
        return False

    # Set ngrok auth token
    try:
        from pyngrok import ngrok
        ngrok.set_auth_token(auth_token)
        print(" Ngrok authentication configured")
        return True
    except Exception as e:
        print(f" Failed to configure ngrok: {e}")
        return False

def start_streamlit_app():
    """Start the Streamlit dashboard in the background."""
    print(" Starting Streamlit dashboard...")

    # Import here to avoid issues if streamlit isn't installed
    try:
        import streamlit.web.cli as st_cli
    except ImportError:
        print(" Streamlit not installed. Run: pip install -r requirements-py314.txt")
        return None

    # Path to the app
    app_path = Path(__file__).parent / "src" / "phase4_deploy" / "app.py"

    if not app_path.exists():
        print(f" Streamlit app not found at {app_path}")
        return None

    # Start streamlit in background
    try:
        # Use subprocess to run streamlit
        cmd = [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", "8501", "--server.headless", "true"]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(" Streamlit app started on port 8501")
        return process
    except Exception as e:
        print(f" Failed to start Streamlit: {e}")
        return None

def create_ngrok_tunnel():
    """Create ngrok tunnel to the Streamlit app."""
    try:
        from pyngrok import ngrok

        # Create tunnel
        tunnel = ngrok.connect(8501, "http")
        public_url = tunnel.public_url

        print(" Ngrok tunnel created!")
        print(f"   Public URL: {public_url}")
        print(f"   Local URL:  http://localhost:8501")
        print("\n Share this link for your demo:")
        print(f"   {public_url}")
        print("\n Tunnel will be active for 8 hours (ngrok free tier limit)")

        return tunnel, public_url

    except Exception as e:
        print(f" Failed to create ngrok tunnel: {e}")
        return None, None

def main():
    """Main demo function."""
    from dotenv import load_dotenv
    load_dotenv(override=True)
    
    print("=" * 60)
    print("URBAN RESILIENCE ENGINE - DEMO LAUNCHER")
    print("=" * 60)
    print("\nChoose deployment mode:")
    print("1. Ngrok (public URL, shareable)")
    print("2. Localhost (local XAMPP, development)")
    
    mode = input("\nSelect mode (1/2): ").strip()
    
    # Start Streamlit app
    streamlit_process = start_streamlit_app()
    if not streamlit_process:
        return

    # Give streamlit time to start
    print(" Waiting for Streamlit to initialize...")
    time.sleep(5)
    
    if mode == "1":
        # Setup ngrok
        if not setup_ngrok():
            streamlit_process.terminate()
            return

        # Create ngrok tunnel
        tunnel, public_url = create_ngrok_tunnel()
        if not tunnel:
            streamlit_process.terminate()
            return

        print("\n" + "=" * 60)
        print(" DEMO READY - NGROK MODE!")
        print("=" * 60)
        print(f"Public URL:  {public_url}")
        print(f"Local URL:   http://localhost:8501")
        print("\n Tunnel active for 8 hours (ngrok free tier limit)")
        print("Press Ctrl+C to stop the demo")

        try:
            # Keep running until user stops
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n Stopping demo...")

        # Cleanup ngrok
        if tunnel:
            from pyngrok import ngrok
            ngrok.disconnect(tunnel.public_url)
            print(" Ngrok tunnel closed")
    
    else:
        # Localhost mode
        localhost_url = os.getenv("STREAMLIT_LOCAL_URL", "http://localhost:8501")
        
        print("\n" + "=" * 60)
        print(" DEMO READY - LOCALHOST MODE!")
        print("=" * 60)
        print(f"Dashboard URL: {localhost_url}")
        print("\n XAMPP Integration:")
        print(f"   Apache:     {os.getenv('LOCALHOST_APACHE_URL')}")
        print(f"   MySQL:      {os.getenv('LOCALHOST_MYSQL_HOST')}:{os.getenv('LOCALHOST_MYSQL_PORT')}")
        print(f"   phpMyAdmin: {os.getenv('LOCALHOST_PHPMYADMIN_URL')}")
        print("\nPress Ctrl+C to stop the demo")
        
        try:
            # Keep running until user stops
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n Stopping demo...")

    # Cleanup streamlit
    if streamlit_process:
        streamlit_process.terminate()
        print(" Streamlit app stopped")

    print(" Demo ended. Thanks for using Urban Resilience Engine!")

if __name__ == "__main__":
    main()