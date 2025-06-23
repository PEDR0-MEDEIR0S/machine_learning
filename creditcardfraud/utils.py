import importlib
import subprocess
import sys


class DependencyManager:
    def __init__(self):
        self.installed = set()
        
    def auto_import(self, package_name, import_as=None):
        try:
            module = importlib.import_module(import_as or package_name)
        except ImportError:
            base_package = package_name.split('.')[0]
            if base_package not in self.installed:
                print(f'[DependencyManager] Instalando {base_package}')
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', base_package])
                self.installed.add(base_package)
                print(f'[DependencyManager] Pacote {base_package} instalado.')
            module = importlib.import_module(import_as or package_name)
        return module
