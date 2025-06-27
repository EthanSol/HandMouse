import os
import subprocess
import sys

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(PROJECT_DIR, 'venv')
REQUIREMENTS = os.path.join(PROJECT_DIR, 'requirements.txt')

def create_venv():
    print("Creating virtual environment in ./venv ...")
    subprocess.check_call([sys.executable, '-m', 'venv', VENV_DIR])
    print("Virtual environment created.")


def install_requirements(python_exec):
    print(f"Installing dependencies from {REQUIREMENTS} ...")
    subprocess.check_call([python_exec, '-m', 'pip', 'install', '--upgrade', 'pip'])
    subprocess.check_call([python_exec, '-m', 'pip', 'install', '-r', REQUIREMENTS])
    print("All dependencies installed.")


def main():
    use_venv = input("Do you want to create and use a virtual environment for this project? (y/n): ").strip().lower()
    if use_venv == 'y':
        if not os.path.exists(VENV_DIR):
            create_venv()

        # Get path to virtual environment's Python executable
        binary_dir = 'Scripts' if os.name == 'nt' else 'bin'
        python_executable = 'python.exe' if os.name == 'nt' else 'python'
        python_exec = os.path.join(VENV_DIR, binary_dir, python_executable)

        install_requirements(python_exec)
        print(f"\nTo activate your virtual environment, run:\n  source {os.path.relpath(VENV_DIR)}/bin/activate\n")
    else:
        install_requirements(sys.executable)

if __name__ == "__main__":
    main()
