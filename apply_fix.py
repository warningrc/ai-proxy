import sys

with open('admin_reloader.py', 'r') as f:
    content = f.read()

# Make sure we don't have import main anymore
if 'import main' in content:
    print("Found 'import main' in admin_reloader.py, please remove it.")
