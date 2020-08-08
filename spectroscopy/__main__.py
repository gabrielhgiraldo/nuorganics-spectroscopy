from spectroscopy.app import app
from threading import Timer
import webbrowser

def main():
    port = 8050
    Timer(1, lambda: webbrowser.open_new(f"http://localhost:{port}")).start()
    app.run_server(port=port)

if __name__ == "__main__":
    main()