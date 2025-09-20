# run_app.py
import os, sys, subprocess, time, webbrowser
import socket

# ✅ PyInstaller 打包后解压路径处理
if getattr(sys, 'frozen', False):
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MONGO_BIN = os.path.join(BASE_DIR, "mongodb", "bin", "mongod")

def start_mongo():
    data_dir = os.path.join(BASE_DIR, "mongodb", "data")
    log_file = os.path.join(BASE_DIR, "mongodb", "mongo.log")
    os.makedirs(data_dir, exist_ok=True)

    mongo_cmd = [
        MONGO_BIN,
        "--dbpath", data_dir,
        "--port", "27017",
        "--bind_ip", "127.0.0.1"
    ]

    with open(log_file, "w") as f:
        subprocess.Popen(mongo_cmd, stdout=f, stderr=f)


def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0

def start_streamlit():
    app_file = os.path.join(BASE_DIR, "app.py")
    if not is_port_in_use(8501):
        streamlit_bin = shutil.which("streamlit")
        cmd = [streamlit_bin, "run", app_file, "--server.port", "8501"]
        subprocess.Popen(cmd)
        time.sleep(5)
    webbrowser.open("http://localhost:8501")


if __name__ == "__main__":
    print("🚀 Starting MongoDB...")
    start_mongo()
    time.sleep(3)

    print("📊 Launching Streamlit Dashboard...")
    start_streamlit()
